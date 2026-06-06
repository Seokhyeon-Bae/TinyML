# LCTES IoT 논문용 초안: 모델 3종 + QAT 인용

v2_PGD run (2026-03-16_06-12-39, CIC-IDS2017, FL+압축+PGD) 결과 기준.

---

## 1. 권장 모델 세 가지 (실험 결과 요약)

실험에서 압축 파이프라인을 2×2로 설계하였다. 학습 시 QAT 사용 여부와 압축 시 양자화 방식(PTQ vs QAT 파인튜닝)을 조합한 네 가지 압축 스테이지와, 압축 없는 Baseline을 평가하였다. 그 결과를 바탕으로 **세 가지 용도별 권장 모델**을 제안한다.

| 용도 | 권장 모델 | 설명 | 수치 (v2_PGD run) |
|------|-----------|------|-------------------|
| **성능·크기 균형 (일반 배포)** | **QAT+QAT** | FL에서 QAT로 학습한 뒤, 프루닝 후 다시 QAT 파인튜닝하여 int8로 양자화. F1과 모델 크기 균형이 가장 좋음. | F1 **0.9035**, 크기 **0.167 MB**, 압축률 **4.66×** (78.6% 감소) |
| **PGD 대비 견고성** | **QAT+PTQ** | FL에서 QAT로 학습한 뒤, 프루닝 후 **PTQ만** 적용 (추가 QAT 파인튜닝 없음). 동일 PGD 공격 하에서 압축 모델 중 방어 성능이 가장 높음. | Adv Acc **0.922**, 공격 성공률 **0.020** (가장 낮음) |
| **압축률 극대화** | **QAT+QAT** 또는 **Traditional+QAT** | 최대 압축률 4.66× (78.6% 감소), 크기 0.167 MB. Traditional+QAT는 FL 단계에서는 QAT를 쓰지 않고, 압축 단계에서만 QAT 파인튜닝을 적용한 경우로, 동일 압축률을 달성함. | 압축률 **4.66×**, 크기 **0.167 MB** |

- **일반 배포(정상 80% 수준)**를 가정할 때: ratio sweep 상 80% normal 구간에서 F1이 높은 **QAT+QAT** 또는 **pruned_5x10_qat** 계열을 사용하는 것이 적합하다.
- **PGD 공격이 우려되는 환경**에서는 **QAT+PTQ**를 선택하면, 압축 후에도 adversarial accuracy를 가장 잘 유지한다.
- **용량 제약이 극심한 엣지**에서는 **압축률 4.66×**를 만족하는 QAT+QAT 또는 Traditional+QAT를 사용할 수 있다.

---

## 2. “학습 중 QAT가 항상 유리한 것은 아니다” — 논문 인용용 문단

Quantization-aware training (QAT)은 양자화를 고려한 학습으로, 흔히 post-training quantization (PTQ)보다 낮은 정밀도에서도 성능이 좋은 것으로 알려져 있다. 그러나 **학습 단계에서 QAT를 사용하는 것이 항상 최선은 아니다**는 점이 선행 연구와 본 실험 모두에서 확인된다.

선행 연구에서는, QAT 과정에서 양자화 가중치가 두 그리드 점 사이를 오가는 **진동(oscillation)**이 발생할 수 있으며, 이로 인해 추론 시 batch normalization 통계가 잘못 추정되고 학습 불안정으로 정확도가 떨어질 수 있음이 보고되었다 [1]. 특히 낮은 비트(≤4비트)와 효율적인 구조(MobileNet, EfficientNet 등)에서 두드러진다. 또한 8비트 수준에서는 **PTQ만으로도 부동소수점에 가까운 정확도를 얻는 경우가 많다**는 정리도 있다 [2]. 즉, 작업·비트폭·구조에 따라 PTQ가 QAT에 버금가거나, QAT가 오히려 불리할 수 있다.

본 실험에서도 이와 일치하는 결과가 나왔다. **PGD 공격 하에서 가장 견고한 압축 모델은 QAT+PTQ**였으며, 프루닝 후 QAT 파인튜닝을 한 QAT+QAT보다 adversarial accuracy가 더 높았다 (Adv Acc 0.922 vs 0.883). 반면 **Traditional+QAT**(FL에서는 QAT 없이 학습, 압축 단계에서만 QAT 파인튜닝)는 일반 F1과 압축률(4.66×) 측면에서는 QAT+QAT와 비슷한 수준을 보였으나, PGD에 대해서는 취약하였다 (Adv Acc 0.181). 이는 “학습 중 QAT 사용 여부”와 “압축 후 PTQ vs QAT” 조합에 따라 성능·견고성이 달라짐을 보여 주며, **QAT를 학습에 항상 넣는 것이 최선이 아님**을 실험적으로 뒷받침한다.

**[1]** M. Nagel, M. Fournarakis, Y. Bondarenko, and T. Blankevoort, “Overcoming oscillations in quantization-aware training,” in *Proc. ICML*, 2022, pp. 16318–16330.

**[2]** M. Nagel et al., “A white paper on neural network quantization,” arXiv:2106.08295, 2021. (PTQ가 8비트에서 대부분 충분하다는 정리 포함.)

---

## 3. BibTeX (참고)

```bibtex
@inproceedings{nagel2022overcoming,
  title     = {Overcoming Oscillations in Quantization-Aware Training},
  author    = {Nagel, Markus and Fournarakis, Marios and Bondarenko, Yelysei and Blankevoort, Tijmen},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {162},
  pages     = {16318--16330},
  year      = {2022},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v162/nagel22a.html},
}

@article{nagel2021white,
  title   = {A White Paper on Neural Network Quantization},
  author  = {Nagel, Markus and others},
  journal = {arXiv preprint arXiv:2106.08295},
  year    = {2021},
}
```

---

## 4. 논문에 넣을 때 참고

- **표**: 위 표를 논문 스타일에 맞게 “Recommended models for deployment” 같은 캡션으로 넣고, 필요 시 F1/Adv Acc/압축률만 요약한 더 짧은 버전으로 줄여도 됨.
- **인용 문단**: “Related work / Discussion” 또는 “Quantization strategy” 부분에 2절 문단을 넣고, [1][2]를 참고문헌에 추가하면 됨.
- **수치**: 모두 v2_PGD run (2026-03-16_06-12-39) 기준이므로, 논문에는 “our experiment (v2_PGD)” 또는 “CIC-IDS2017, FL 70 rounds, …” 정도의 실험 설정 한 줄을 함께 넣으면 좋음.

이 초안을 LCTES IoT PDF에 맞게 문장만 다듬어서 사용하시면 됩니다.
