# 在 VS Code 中查看对照页

预览地址：http://127.0.0.1:8765/demonstrations/index.html

## 推荐入口

1. 在“运行和调试”面板选择 **ContextFlow: 示范与评估视频对照**，按 F5。
2. 该配置先执行本地预览启动任务，随后在 VS Code 的内置浏览器标签中打开页面。服务已运行时会复用。关闭调试页不会停止预览服务，便于再次打开。

也可按 Ctrl+Shift+P，运行 **Browser: Open Integrated Browser**，输入上面的地址。不要把集群 HTML 文件路径直接当成本地浏览器的 file URL。

## Remote SSH 连接

内置浏览器支持通过远程连接访问集群 localhost。项目附带了 enableRemoteProxy=true、dataStorage=workspace 的本地 VS Code 设置。若当前使用多根工作区、文件夹设置未生效，或仍提示连接被拒绝：在 VS Code 的“端口 / Ports”面板添加转发 **8765**，再使用面板显示的本地地址（本地端口被占用时可能不同）。也可在工作区级设置启用 `workbench.browser.enableRemoteProxy` 并把 `workbench.browser.dataStorage` 设为 `workspace`。

## 手动启动

在 ContextFlow 根目录运行：

```bash
.venv/bin/python scripts/serve_goal_long_viewer.py --ensure
```

服务仅监听集群 127.0.0.1:8765，提供本次实验目录；不开放目录列表。支持 MP4 的 Range/HEAD 和正确 MIME，方便加载及拖动进度。源视频保持 H.264/yuv420p，未转码或改动。服务 PID、日志分别在实验目录的 viewer_service.json、viewer_http.log。

## 已验证与限制

HTTP 页面可访问，80 个页面资源 HEAD 成功，60 个视频完成头部/尾部 Range 核对；两个重点示范均通过 HTTP 实际解码。不能从这里操作或验证用户本地 VS Code GUI，实际窗口打开与播放尚需在客户端确认。记录见 ../viewer_validation.json。

官方说明：[内置浏览器](https://code.visualstudio.com/docs/debugtest/integrated-browser)，包括 editor-browser 启动配置及 Remote SSH 代理。
