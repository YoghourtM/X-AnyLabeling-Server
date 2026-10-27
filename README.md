<div align="center">
  <p>
    <a href="https://github.com/CVHub520/X-AnyLabeling/" target="_blank">
      <img alt="X-AnyLabeling" height="200px" src="https://github.com/user-attachments/assets/0714a182-92bd-4b47-b48d-1c5d7c225176"></a>
  </p>

  <h3 align="center">
    Simple, Lightweight, and Extensible Serving for X-AnyLabeling
  </h3>

</div>

<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)

<img alt="X-AnyLabeling-Server Logo" src="assets/logo/X-AnyLabeling-Server.png" style="width:100%;display:block;" />

## About this fork

This fork of [CVHub520/X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server) adds a local **Segment Anything 3.1** service for the [X-AnyLabeling fork](https://github.com/YoghourtM/X-AnyLabeling/tree/feat/vlm-annotation): text prompts, clicks / a box, and example boxes from one `sam3.1_multiplex.pt` checkpoint in about 5.5 GB of GPU memory, run in docker on `127.0.0.1:8000`.

```bash
export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
scripts/sam31_server.sh build && scripts/sam31_server.sh start
```

Details: [docker/sam31/README.md](./docker/sam31/README.md). Client-side usage: [English](https://github.com/YoghourtM/X-AnyLabeling/blob/feat/vlm-annotation/docs/en/vlm_annotator.md) | [中文](https://github.com/YoghourtM/X-AnyLabeling/blob/feat/vlm-annotation/docs/zh_cn/vlm_annotator.md). The rest of this README is upstream's.

本仓库 fork 自 CVHub520/X-AnyLabeling-Server，新增本地 SAM 3.1 服务（文本提示、点击或框选、示例框找相似），中文使用说明见上方链接。

## About

X-AnyLabeling-Server is a simple, lightweight and extensible serving framework for AI model inference, specifically designed for [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling). It provides a production-ready solution with pluggable architecture and flexible configuration for various auto-labeling scenarios. Its core features include:

- **Decoupled Design**: Framework handles service management and resource scheduling without interfering with model implementation details
- **Pluggable Architecture**: Rapidly integrate custom models without modifying core framework code
- **Production Ready**: Comprehensive structured logging, error handling, concurrency control, and security authentication
- **Flexible Configuration**: All parameters are configurable with sensible defaults, adaptable to different deployment scenarios

## Getting Started

- [Installation and Setup](./docs/source/en/get_started.md)
- [Configuration Guide](./docs/source/en/configuration.md)
- [Custom Model Integration](./docs/source/en/user_guide.md)
- [API Guide](./docs/source/en/router.md)
- [OpenAPI Schema](./docs/openapi.json)

## Contributing

Contributions and collaborations are highly appreciated! For guidelines on how to contribute, please refer to [Contributing to X-AnyLabeling-Server](./CONTRIBUTING.md).

## Sponsors

- [buy-me-a-coffee](https://ko-fi.com/cvhub520)
- [Wechat/Alipay](https://github.com/CVHub520/X-AnyLabeling/blob/main/README_zh-CN.md#%E8%B5%9E%E5%8A%A9)

## Citing

If you use this software in your research, please cite it as below:

```
@misc{X-AnyLabeling-Server,
  year = {2025},
  author = {Wei Wang},
  publisher = {Github},
  organization = {CVHub},
  journal = {Github repository},
  title = {A Simple, Lightweight, and Extensible Serving Framework for X-AnyLabeling},
  howpublished = {\url{https://github.com/CVHub520/X-AnyLabeling-Server}}
}

@misc{X-AnyLabeling,
  year = {2023},
  author = {Wei Wang},
  publisher = {Github},
  organization = {CVHub},
  journal = {Github repository},
  title = {X-AnyLabeling: A Unified Desktop Platform for AI-Assisted Data Annotation
},
  howpublished = {\url{https://github.com/CVHub520/X-AnyLabeling}}
}
```

---

![Star History Chart](https://api.star-history.com/svg?repos=CVHub520/X-AnyLabeling-Server&type=Date)

<div align="center"><a href="#top">🔝 Back to Top</a></div>
