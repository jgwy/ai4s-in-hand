import { defineConfig } from 'vitepress'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/ai4s-in-hand/'

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI4S in Hand',
  description: '从科学问题到自主发现的 AI for Science 实践教程',
  base: baseConfig,
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    math: true
  },
  themeConfig: {
    logo: '/datawhale-logo.png',
    nav: [
      { text: '学习地图', link: '/preface/' },
      {
        text: '双入口',
        items: [
          { text: 'Science 背景：AI 补给', link: '/part1-ai-foundations/' },
          { text: 'AI 背景：Science 补给', link: '/part2-science-foundations/' },
          { text: '共同合流：科研新范式', link: '/part3-research-paradigm/' }
        ]
      },
      { text: '综合项目', link: '/projects/' }
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    sidebar: [
      {
        text: '开始',
        items: [
          { text: '项目首页', link: '/' },
          { text: '第 0 章：AI4S 学习地图', link: '/preface/' }
        ]
      },
      {
        text: '第一部分：AI 补给',
        items: [
          { text: '部分导读与章节地图', link: '/part1-ai-foundations/' },
          { text: '第 1 章：科学问题与学习任务', link: '/part1-ai-foundations/chapter1' },
          { text: '第 2 章：科学数据与可信基线', link: '/part1-ai-foundations/chapter2' },
          { text: '第 3 章：科学结构深度学习', link: '/part1-ai-foundations/chapter3' },
          { text: '第 4 章：科学机器学习', link: '/part1-ai-foundations/chapter4' },
          { text: '第 5 章：从预测到决策', link: '/part1-ai-foundations/chapter5' }
        ]
      },
      {
        text: '第二部分：Science 补给',
        items: [
          { text: '部分导读与章节地图', link: '/part2-science-foundations/' },
          { text: '第 6 章：科学方法与可信验证', link: '/part2-science-foundations/chapter6' },
          { text: '第 7 章：物理与计算科学', link: '/part2-science-foundations/chapter7' },
          { text: '第 8 章：化学与分子系统', link: '/part2-science-foundations/chapter8' },
          { text: '第 9 章：材料科学', link: '/part2-science-foundations/chapter9' },
          { text: '第 10 章：生命科学', link: '/part2-science-foundations/chapter10' },
          { text: '第 11 章：经典 AI4S', link: '/part2-science-foundations/chapter11' }
        ]
      },
      {
        text: '第三部分：科研新范式',
        items: [
          { text: '部分导读与章节地图', link: '/part3-research-paradigm/' },
          { text: '第 12 章：科研知识与文献智能', link: '/part3-research-paradigm/chapter12' },
          { text: '第 13 章：提出研究问题', link: '/part3-research-paradigm/chapter13' },
          { text: '第 14 章：AI 辅助计算研究', link: '/part3-research-paradigm/chapter14' },
          { text: '第 15 章：高通量与自动化实验室', link: '/part3-research-paradigm/chapter15' },
          { text: '第 16 章：自主发现闭环', link: '/part3-research-paradigm/chapter16' },
          { text: '第 17 章：写作、评审与传播', link: '/part3-research-paradigm/chapter17' },
          { text: '第 18 章：可信与负责任 AI4S', link: '/part3-research-paradigm/chapter18' },
          { text: '第 19 章：综合项目', link: '/part3-research-paradigm/chapter19' }
        ]
      },
      {
        text: '项目与规范',
        items: [
          { text: '综合项目', link: '/projects/' },
          { text: '写作指南', link: '/appendices/writing-guide' },
          { text: '引用与事实核验', link: '/appendices/references' },
          { text: '可复现性规范', link: '/appendices/reproducibility' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/jgwy/ai4s-in-hand' }
    ],
    editLink: {
      pattern: 'https://github.com/jgwy/ai4s-in-hand/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },
    lastUpdated: {
      text: '最后更新于'
    },
    outline: {
      label: '本页目录',
      level: [2, 3]
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    footer: {
      message: '本教程仍在持续核验与完善中。',
      copyright: '采用 CC BY-NC-SA 4.0 许可协议'
    }
  }
})
