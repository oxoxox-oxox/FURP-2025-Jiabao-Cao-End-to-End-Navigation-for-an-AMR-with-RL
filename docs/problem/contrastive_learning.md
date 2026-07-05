# Contrastive Learning + End-to-End RL Visual Navigation

## 1. Motivation

为什么值得做？

- 视觉导航容易受到光照、天气、曝光等外观变化影响，导致导航性能下降
- 当前主要依赖 Data Augmentation 或 Domain Randomization，提高鲁棒性依赖大量训练数据
- 模拟器可以方便生成同一场景在不同光照条件下的数据，为 Contrastive Learning 提供天然的正样本对
- 希望利用额外监督学习外观不变的导航特征，提高泛化能力

---

## 2. Core Idea

你打算怎么做？

- 在模拟器中固定机器人 Pose
- 随机改变光照、曝光、天气等环境条件，生成同一位置的多张图像
- 利用 Contrastive Learning 拉近不同外观下的 Navigation Feature
- 与 End-to-End RL联合训练（如果可以的话），使策略更加鲁棒

---

## 3. Research Gap

别人没有解决什么？

- 现有视觉导航主要依赖数据增强，没有显式约束相同场景应具有一致的导航表示
- Contrastive Learning 多用于分类、定位、Place Recognition，很少应用于 End-to-End RL Navigation
- 模拟器能够提供精确 Pose 信息，但很少利用其自动构造 Positive Pair
- 所有类似 CLIP backbone 提供的都是 semantic feature，但是导航需要的是空间几何性质的 alignment

---

## 4. Novelty

真正的创新点是什么？

1. 利用模拟器自动生成同一 Pose、不同外观条件下的 Positive Pair
2. 提出面向导航任务的 Appearance-Invariant Navigation Representation，而非普通视觉特征学习
3. Contrastive Learning 与 RL 联合优化（可能），提高导航策略在未知条件下的泛化能力

---

## 5. Related Work

Paper:

- End-to-End Visual Navigation（DD-PPO、ViNT、NoMaD）
- Contrastive Learning（SimCLR、MoCo、BYOL、DINO）
- Visual Place Recognition（NetVLAD、CosPlace）
- Domain Randomization

Difference:

已有方法主要提升视觉特征或增加数据多样性，缺少针对导航表示的一致性约束

---

## 6. Feasibility

| | |
| --------- | ----------------------------------- |
| Data | 模拟器在线生成 |
| Simulator | Isaac Sim、Habitat、Gazebo |
| Code | 对现有 RL 框架修改 Encoder |
| GPU | RTX 3060 + |
| Difficulty | 2 |

---

## 7. Expected Contribution

- 理论贡献：提出 Appearance-Invariant Navigation Representation 的学习框架

- 工程贡献：提供一种轻量级、易集成的视觉导航鲁棒性提升方法

- 实验贡献：验证在未知光照条件下，导航成功率、稳定性和泛化能力均有所提升

---

## 8. Risks

1. Contrastive Learning 相比 Data Augmentation 提升有限
2. Feature 更稳定，但导航性能提升不明显
3. 创新点只是加入 Contrastive Loss，需要突出导航任务导向

---

## 9. Next Step

- 阅读视觉导航与 Contrastive Learning 相关论文
- 复现一个 Visual End-to-End RL Navigation Baseline
- 在模拟器中实现光照随机化
- 构建 Positive Pair 并加入 Contrastive Loss
- 与 Data Augmentation、Domain Randomization 进行比较实验
- 完成消融实验，分析不同 Loss 和不同外观变化的影响

## 10. Reviewer Questions

- Q1 为什么不用 Data Augmentation？

- Q2 为什么不用 CLIP、DINO 等预训练模型？

- Q3 Contrastive Learning 为什么能够提升导航，而不仅是 Feature？

- Q4 创新点是否只是增加了一个 Contrastive Loss？

- Q5 为什么选择在 Navigation Feature 而不是 Backbone Feature 上进行约束？

- Q6 为什么只在模拟器训练能够推广到真实机器人？

- Q7 如何证明表示一致性最终带来了策略一致性和导航性能提升？
