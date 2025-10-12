# 🌍 UI-Simulator

[![License](https://img.shields.io/github/license/WadeYin9712/UI-Simulator)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Stars](https://img.shields.io/github/stars/WadeYin9712/UI-Simulator?style=social)]()

🔗 **Paper link**: [Link]() 

🌐 **Website**: https://ui-simulator.notion.site/llms-as-scalable-digital-world-simulator

🤗 **Model Weights:** [Link]()  

📚 **Datasets:** [Link]()  

📧 **Contact:** [da.yin9712@gmail.com](mailto:da.yin9712@gmail.com), [w10y20ming@gmail.com](mailto:w10y20ming@gmail.com)

---

## 🚀 Overview

**🌍 UI-Simulator** is a scalable, general-purpose simulator designed for **digital agent training** — and even applications **beyond digital environments**.  
It provides a robust and extensible simulation framework to enable efficient, data-rich interaction modeling.

![UI-Simulator Overview](https://github.com/WadeYin9712/UI-Simulator/tree/main/figures/uisimulator_intro.png)

---

## ✨ Key Features

- 🔧 **Scalable & General-Purpose:** Generalized design adaptable to multiple digital environments, or even beyond.  
- ⚙️ **Targeted Scaling Strategy:** Enables more rapid and **data-efficient** agent training.  
- 🧩 **Comprehensive Evaluation:** Demonstrated strong results on **WebArena** and **AndroidWorld** benchmarks.  

---

## 🧩 Installation

```bash
git clone https://github.com/WadeYin9712/UI-Simulator.git
cd UI-Simulator

conda create --name ui_simulator python=3.11
conda activate ui_simulator
pip install -r requirements.txt
```
If you wanna run evaluation on **WebArena** or **AndroidWorld**, please configure the venv according to their official installation guide. We refer the readers to 

---

## 🧪 Running

### 📊 Data Collection

We provides the shells for running the data collection [here](https://github.com/WadeYin9712/UI-Simulator/shells)

```bash
export OPENAI_API_KEY=<YOUR_KEY>
export OPENAI_ORG_ID=<YOUR_ORG_ID>

# e.g. running rag-based data collection on web env, in gitlab domain
bash shells/web_collector/rag/gitlab.sh
```

---

## 📚 Citation

If you find this work useful, please consider citing:
[bibtex]



🧭 Automatic Contributors
<a href="https://github.com/WadeYin9712/UI-Simulator/graphs/contributors"> <img src="https://contrib.rocks/image?repo=WadeYin9712/UI-Simulator" /> </a>
