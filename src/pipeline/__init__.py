# Pipeline orchestration scripts for the Asymmetric Dual-Pass Reflection VAD system.
#
# Each stage runs as an independent process. GPU hosts only ONE model at a time.
#
#   Stage A [VLM]: Coarse blind captioning  (interval=16)
#   Stage B [LLM]: Initial scoring
#   Stage C [LLM]: Context memory + Conflict detection  (Phase 2 + Phase 3)
#   Stage D [VLM]: Targeted fine-grained verification  (Phase 4, interval=4)
#   Stage E [LLM]: Final scoring + merge + Gaussian smooth
