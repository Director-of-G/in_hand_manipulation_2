# Robust Model-Based In-Hand Manipulation <br> with Integrated Real-Time Motion-Contact <br> Planning and Tracking

<p style="text-align: center;"> 
submitted to the <b> International Journal of Robotics Research (IJRR) </b>, 2025
</p>

<p style="text-align: center;"> 
Yongpeng Jiang, Mingrui Yu, Xinghao Zhu, Masayoshi Tomizuka and Xiang Li
</p>

<p style="text-align: center;"> 
Tsinghua University
</p>

<p style="text-align: center;"> 
<a href="https://arxiv.org/abs/2505.04978" style="color: #0ABAB5; text-decoration: underline;">arXiv</a> |
<a href="https://github.com/Director-of-G/in_hand_manipulation_2" style="color: #0ABAB5; text-decoration: underline;">Code</a> |
<a href="https://youtu.be/vppT66jVsGo?si=G3ZnwMo6oex3u9oq" style="color: #0ABAB5; text-decoration: underline;">Video</a>
</p>

## Video

<iframe width="900" height="506" src="https://www.youtube.com/embed/vppT66jVsGo?si=i9IEZhhLijSOLnv7" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Hardware Performance

<div align="center">
  <img src="./media/hardware_teaser.gif" alt="Hardware Performance" width="75%" />
</div>

## Abstract

Robotic dexterous in-hand manipulation, where multiple fingers dynamically make and break contact, represents a step toward human-like dexterity in real-world robotic applications. Unlike learning-based approaches that rely on large-scale training or extensive data collection for each specific task, model-based methods offer an efficient alternative. Their online computing nature allows for ready application to new tasks without extensive retraining. However, due to the complexity of physical contacts, existing model-based methods encounter challenges in efficient online planning and handling modeling errors, which limit their practical applications. To advance the effectiveness and robustness of model-based contact-rich in-hand manipulation, this paper proposes a novel integrated framework that mitigates these limitations. The integration involves two key aspects: 1) integrated real-time planning and tracking achieved by a hierarchical structure; and 2) joint optimization of motions and contacts achieved by integrated motion-contact modeling. Specifically, at the high level, finger motion and contact force references are jointly generated using contact-implicit model predictive control. The high-level module facilitates real-time planning and disturbance recovery. At the low level, these integrated references are concurrently tracked using a hand force-motion model and actual tactile feedback. The low-level module compensates for modeling errors and enhances the robustness of manipulation. Extensive experiments demonstrate that our approach outperforms existing model-based methods in terms of accuracy, robustness, and real-time performance. Our method successfully completes five challenging tasks in real-world environments, even under appreciable external disturbances.

## Citation

Please cite our paper if you find it helpful :)

```
misc{jiang2025robustmodelbasedinhandmanipulation,
      title={Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking}, 
      author={Yongpeng Jiang and Mingrui Yu and Xinghao Zhu and Masayoshi Tomizuka and Xiang Li},
      year={2025},
      eprint={2505.04978},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.04978}, 
}
```

## Contact

If you have any question, feel free to contact the authors: Yongpeng Jiang, [jiangyp19@gmail.com](mailto:jiangyp19@gmail.com) .

Yongpeng Jiang's Homepage is at [director-of-g.github.io](https://director-of-g.github.io/).
