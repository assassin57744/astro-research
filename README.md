# astro-research 🔭

> A dedicated academic portfolio space focusing on astrophysical research, self-developed open-source tools, astrophotography, and science communication.

---

## 📌 Overview

This repository serves as a centralized hub to document my scientific research practices and engineering explorations as an astronomy enthusiast. The content is structured into five core modules:

* **🧬 Research Projects**: Focuses on data analysis of specific celestial objects and catalogs. The current flagship project is the **Hunt 2024 Star Catalog Auditing**.
* **🛠️ Tools**: Houses general automation and data processing scripts developed during my research and daily observations to improve workflow efficiency (e.g., **astro_data_downloader**).
* **🎨 Astrophotography**: Showcases deep-sky, planetary, and lunar astrophotography work, including post-processing stacks and detailed imaging logs.
* **🔭 Observation Logs**: Documents raw visual observation data, star chart comparisons, and telescope equipment calibration notes.
* **📝 Science Communication**: Contains literature outlines for Wikipedia astronomy article contributions, alongside open-source resources for school clubs and astronomy competitions.

---

## 📂 Directory Structure

```text
├── /research/                 # 📌 Research Projects
│   └── /hunt24-audit/         # Hunt 2024 star catalog auditing project
│       ├── README.md          # Project background, search radius verification, & summary
│       └── cluster_filter.py  # Data cleaning algorithms for specific clusters (e.g., Hyades)
├── /tools/                    # 🛠️ Self-Developed Tools
│   └── astro_data_downloader.py  # Automated downloader for astronomical catalogs (VizieR/Gaia)
├── /astrophotography/         # 🎨 Astrophotography Gallery
│   ├── /gallery/              # Curated astrophotos (JPEG format optimized for web)
│   └── imaging_logs.md        # Technical logs (mount, camera, exposure specs, & post-processing)
├── /logs/                     # 🔭 Raw Observation Logs
├── /wiki-drafts/              # 📝 Wikipedia Drafts & Outlines
└── /club-resources/           # 🎪 School Astronomy Club & Competition Resources
```
---

## 🎯 Current Focus

### 1. Project: Hunt 2024 Catalog Auditing (`/research/hunt24-audit/`)
* **Objective**: Cross-verify and audit member stars within the Hunt 2024 catalog using open astronomical big data.
* **Core Tasks**: Utilize the latest satellite observation datasets to validate the search radii of specific star clusters (such as the Hyades cluster), filtering out background noise to ensure catalog precision.

### 2. Tool: `astro_data_downloader` (`/tools/`)
* **Objective**: A self-developed utility script designed for batch retrieving astronomical data.
* **Problem Solved**: Eliminates the friction of manual data fetching by automating stable, programmatic queries from academic databases like VizieR.

### 3. Astrophotography: Deep-Sky & Planetary/Lunar Exploration (`/astrophotography/`)
* **Objective**: Calibrating deep-sky imagery using dark/flat frames, documenting SNR (Signal-to-Noise Ratio) under varying light pollution levels, and merging astronomical science with visual aesthetics.

---

## 🛠️ Roadmap

- [ ] Archive filtering algorithms and 2D transformation visualization plots for the `hunt24-audit` project.
- [ ] Optimize Git repository history by purging heavy, temporary data files (e.g., `.csv`, `.vot`).
- [ ] Upload recent deep-sky/lunar captures and complete `/astrophotography/imaging_logs.md`.
- [ ] Implement robust exception handling and multi-source API support for `astro_data_downloader`.
- [ ] Organize and archive question banks and logistics workflows from recent school astronomy competitions.

---

📬 **Connect**: If you are interested in any of the research, tools, or astrophotography here, or if you share the same passion for science communication, feel free to open an Issue or submit a Pull Request!

---
---

# astro-research (中文版) 🔭

> 这是一个聚焦于天体物理专项研究、自研开源工具、天文摄影与科普的个人学术 Portfolio 空间。

---

## 📌 项目概述

本仓库用于沉淀我作为一名天文爱好者的日常科研实践、工程探索与观测审美。内容主要涵盖以下五个核心板块：

* **🧬 专项研究课题**：聚焦于特定天体的天文大数据分析。目前核心项目为 **Hunt 2024 星表审计**。
* **🛠️ 自研工具箱**：沉淀在研究和日常观测中开发的通用自动化与数据处理脚本，提高科研效率（如 **astro_data_downloader**）。
* **🎨 天文摄影作品**：记录深空（Deep Sky）与行星/月面等天文摄影作品，包含后期堆栈处理流与拍摄参数日志。
* **🔭 观测日志**：记录日常实际望远镜目视观测的实测数据、星图比对及设备调试。
* **📝 科普与词条底稿**：参与维基百科天文词条编写的文献大纲，以及面向校园社团和竞赛组织的开源资料。

---

## 📂 目录结构

```text
├── /research/                 # 📌 专项研究课题
│   └── /hunt24-audit/         # Hunt 2024 星团成员星审计课题
│       ├── README.md          # 课题背景、搜索半径验证方法与结论摘要
│       └── cluster_filter.py  # 针对特定星团（如毕宿星团）的成员星数据清洗算法
├── /tools/                    # 🛠️ 自研通用天文工具箱
│   └── astro_data_downloader.py  # 通用天文数据/星表自动下载器（如对接 VizieR/Gaia 等）
├── /astrophotography/         # 🎨 天文摄影画廊
│   ├── /gallery/              # 存放精选摄影作品（推荐压缩后的 JPEG 格式以节省空间）
│   └── imaging_logs.md        # 拍摄参数日志（记录赤道仪、相机、曝光参数及后处理软件）
├── /logs/                     # 🔭 个人天文观测实测记录
├── /wiki-drafts/              # 📝 维基百科词条修改与编写草稿
└── /club-resources/           # 🎪 校园天文社团活动及竞赛组织的开源资料

```
---

## 🎯 当前专项焦点

### 1. 课题：Hunt 2024 星表审计 (`/research/hunt24-audit/`)
* **研究方向**：基于特定天文大数据，对 Hunt 2024 星表中的星团成员星进行审计与验证。
* **核心工作**：结合最新卫星观测数据，验证特定星团（如 Hyades 毕宿星团等）的搜索半径，通过数据清洗过滤噪声，确保成员星目录的严谨性。

### 2. 工具：`astro_data_downloader` (`/tools/`)
* **功能定位**：自研的通用天文数据下载工具，通过自动化脚本实现高效、稳定的星表数据批量检索与下载，解决手动获取数据的痛点。

### 3. 摄影：深空与地景探索 (`/astrophotography/`)
* **技术路线**：练习深空天体的暗场/平场校准与堆栈，记录不同光害环境下的图像信噪比表现，并尝试将天文技术与美学结合。

---

## 🛠️ 正在进行中

- [ ] 归档 `hunt24-audit` 课题的阶段性过滤算法与 2D 转换可视化图表
- [ ] 优化 Git 仓库历史，清理课题研究中产生的大型临时数据文件（如 .csv / .vot）
- [ ] 整理并上传近期拍摄的精选深空/月面摄影作品，并补齐 `/astrophotography/imaging_logs.md`
- [ ] 完善 `astro_data_downloader` 的异常处理逻辑与多数据源接口
- [ ] 整理近期筹办校园天文竞赛的完整题库与组织流程

---

📬 **交流与合作**：如果你对本仓库中的课题、工具或星空摄影感兴趣，或在天文普及方面有相同的热忱，欢迎提交 Issue 或 Pull Request 一起交流！
