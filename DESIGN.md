自顶向下的感知引导（Top-down Perception Guidance）”或“感知-推理全闭环自适应架构（Closed-loop Adaptive Perception-Reasoning）

这种思路非常契合人类的认知模式：当有人告诉你“附近有着火”，你的眼睛（VLM）就会自动对“亮光、烟雾、红色”等特征赋予更高的敏感度，而不是把它们当成“霓虹灯”或“圣诞树”。

Idea improvement:
1) 离线预提取（Offline Pre-captioning）策略 -> VLM 能够在线接收反馈（Online Feedback-driven），实现“动态注意力转移”dynamic attention
2) 解决VLM的低分辨率问题, 通过将全局上下文作为先验知识（Prior Knowledge）注入 VLM 的 Prompt，可以激发 VLM 捕捉细粒度、任务相关特征的能力。(找找相关论文)
3) 双 Agent 全局-局部反思闭环（Dual-Agent Global-Local Reflection Loop), 将vlm也引入到反思闭环中.

problems:
1) 幻觉:  解决方案: 中立性的提示词
2) 计算开销: 非对称双重反思架构 (Asymmetric Dual-Pass Reflection) llm全量遍历, 但是vlm按需遍历.


视频异常检测 (VAD)：非对称双重反思架构设计方案
一、 核心理念 (Core Concept)
本方案旨在解决当前免训练（Training-free）视频异常检测模型中存在的“局部视野盲区”与“VLM感知缺乏上下文引导”的问题。
为了在“实现全量纠错”与“控制高昂的VLM算力开销”之间取得完美平衡，系统采用非对称解耦设计：

LLM（逻辑大脑）：成本极低，负责执行全量的上下文反思与重评估。

VLM（视觉眼睛）：成本极高，受 LLM 引导，仅执行按需（条件触发）的靶向视觉重采样。

二、 系统工作流 Pipeline 设计 (四阶段)
Phase 1: 局部感知与初判 (Local Perception & Initial Scoring)
执行者：VLM (如 VideoLLaMA) + 主 Agent (LLM)

动作：

VLM 在无先验知识的情况下，按步长遍历视频帧，提取基础特征生成 Caption（例如：“画面中有一棵树在发光”）。

主 Agent (LLM) 基于这个单薄的 Caption 给出初步的异常分数（例如：0.2分，正常）。

输出：包含时间戳、基础 Caption 和初步分数的 Initial_Log。

Phase 2: 全局常态记忆生成 (Global Context Aggregation)
执行者：子 Agent (LLM Summarizer)

动作：

子 Agent 读取一段历史窗口（或高置信度正常帧）的 Initial_Log。

通过 Prompt 提炼出该场景的全局常态背景 (Global Scene Context)（例如：“这是一个夜晚的森林，光线很暗，不应有强光源”）。

输出：scene_context.json (或存入轻量级记忆库)。

Phase 3: 全量文本反思与异常定位 (Textual Reflection & Flagging) —— (非对称设计的核心：低成本全量纠错)
执行者：主 Agent (LLM)

动作：

将 scene_context 注入主 Agent 的 Prompt。

主 Agent 带着全局背景，全量重读第一阶段生成的所有基础 Caption。

主 Agent 进行逻辑推演，寻找“认知冲突”。（例如：LLM发现“森林夜晚”的常态与“发光的树”这个局部描述存在强烈冲突）。

输出：主 Agent 不直接修改分数，而是输出一个“存疑帧列表 (Flagged List)”。只有那些分数发生剧烈波动，或局部描述与全局背景违和的帧才会被标记。

Phase 4: 靶向视觉验证与终判 (Targeted Visual Verification) —— (高成本按需调用)
执行者：VLM + 主 Agent (LLM)

动作：

系统取出“存疑帧列表”，仅针对这些被标记的帧，重新调用 VLM。

自顶向下的感知引导 (Prompt 动态重构)：将全局常态背景和 LLM 的怀疑点结合，生成带有验证性质且中立的 Prompt 喂给 VLM。（例如：“当前场景是夜晚森林，此前检测到‘发光的树’。请仔细核实画面中是否存在火焰、燃烧或异常发光体？”）。

VLM 受到引导，输出更精细的二次 Caption（例如：“树枝正在燃烧，有火焰和烟雾”）。

主 Agent (LLM) 根据二次 Caption 给出最终的修正分数（例如：0.95分，异常）。

输出：最终修正后的异常分数序列。

三、 代码重构/模块化落地指南 (基于 URF-HVAA)
在与 AI 讨论代码实现时，建议按照以下模块进行开发：

Video_Perception_Controller (感知控制器):

需要重构原有的 video_pre_caption.py。

使其支持两种模式：base_caption (无上下文盲搜) 和 guided_caption (接收外部 Prompt 的靶向精搜)。

Context_Memory_Manager (记忆管理器):

基于 summarize_window.py 改造。

负责维护滑动窗口的日志，定期调用 LLM 生成和更新 scene_context。

Asymmetric_Refiner (非对称修正引擎):

这是一个全新的核心控制脚本。

逻辑流：读取 Initial Log -> 注入 Context 调用 LLM 寻找冲突 -> 生成 Flagged List -> 调用感知控制器重新获取 Caption -> 调用 LLM 得出 Final Score。

四、 核心要点提醒 (供开发阶段参考)
防止幻觉 (Anti-Hallucination)：在 Phase 4 中，传给 VLM 的提示词必须是中立/疑问句，绝不能直接说“这里有异常请描述”，防止 VLM 迎合提示词无中生有。

效率监控 (Cost Control)：在代码中设置一个打印信息，对比“全量视频帧数”与“Flagged List 中的帧数”，用以在论文中证明本架构节省了多少 VLM 计算量。