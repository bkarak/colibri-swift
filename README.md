# colibri-swift

A Swift + Metal port of [colibri](https://github.com/JustVugg/colibri) for Apple Silicon:
run mixture-of-experts models from GGUF files, keeping the dense part in RAM and
streaming the routed experts from disk through a cache.

Ported from upstream commit `0a5504a`. Work in progress.
