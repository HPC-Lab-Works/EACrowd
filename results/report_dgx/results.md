## ETH/UCY best-of-20 minADE/minFDE (m)
| Method | ETH | HOTEL | UNIV | ZARA1 | ZARA2 | AVG |
|---|---|---|---|---|---|---|
| Social-GAN (CVPR'18) [published] | 0.87/1.62 | 0.67/1.37 | 0.76/1.52 | 0.35/0.68 | 0.42/0.84 | 0.61/1.21 |
| Social-STGCNN (CVPR'20) [published] | 0.64/1.11 | 0.49/0.85 | 0.44/0.79 | 0.34/0.53 | 0.30/0.48 | 0.44/0.75 |
| Trajectron++ (ECCV'20)* [published] | 0.61/1.03 | 0.20/0.28 | 0.30/0.55 | 0.24/0.41 | 0.18/0.32 | 0.31/0.52 |
| PECNet (ECCV'20) [published] | 0.54/0.87 | 0.18/0.24 | 0.35/0.60 | 0.22/0.39 | 0.17/0.30 | 0.29/0.48 |
| AgentFormer (ICCV'21) [published] | 0.45/0.75 | 0.14/0.22 | 0.25/0.45 | 0.18/0.30 | 0.14/0.24 | 0.23/0.39 |
| MID (CVPR'22)* [published] | 0.55/0.88 | 0.20/0.35 | 0.30/0.55 | 0.29/0.51 | 0.20/0.38 | 0.31/0.53 |
| GroupNet (CVPR'22) [published] | 0.46/0.73 | 0.15/0.25 | 0.26/0.49 | 0.21/0.39 | 0.17/0.33 | 0.25/0.44 |
| MemoNet (CVPR'22) [published] | 0.40/0.61 | 0.11/0.17 | 0.24/0.43 | 0.18/0.32 | 0.14/0.24 | 0.21/0.35 |
| EqMotion (CVPR'23) [published] | 0.40/0.61 | 0.12/0.18 | 0.23/0.43 | 0.18/0.32 | 0.13/0.23 | 0.21/0.35 |
| LED (CVPR'23) [published] | 0.39/0.58 | 0.11/0.17 | 0.26/0.43 | 0.18/0.26 | 0.13/0.22 | 0.21/0.33 |
| MGF (NeurIPS'24) [published] | 0.39/0.59 | 0.13/0.20 | 0.21/0.39 | 0.17/0.29 | 0.14/0.24 | 0.21/0.34 |
| Constant velocity (det.) [ours, this run] | 1.08/2.28 | 0.32/0.61 | 0.52/1.17 | 0.43/0.95 | 0.32/0.72 | 0.53/1.15 |
| Constant velocity + sampling [ours, this run] | 0.72/1.40 | 0.20/0.36 | 0.31/0.61 | 0.26/0.51 | 0.20/0.39 | 0.34/0.65 |
| Social Force Model [ours, this run] | 0.72/1.39 | 0.28/0.43 | 0.51/0.80 | 0.31/0.53 | 0.45/0.67 | 0.45/0.76 |
| LSTM (K-mode) [ours, this run] | 0.51/0.76 | 0.16/0.24 | 0.21/0.38 | 0.19/0.35 | 0.13/0.24 | 0.239±0.001/0.393±0.001 (n=3) |
| Transformer (no RAMG/SGT/MET) [ours, this run] | 0.47/0.69 | 0.14/0.22 | 0.21/0.38 | 0.18/0.33 | 0.13/0.24 | 0.227±0.001/0.374±0.004 (n=3) |
| NSRF-T w/o RAMG [ours, this run] | 0.47/0.71 | 0.13/0.19 | 0.22/0.40 | 0.17/0.31 | 0.13/0.23 | 0.227±0.001/0.368±0.009 (n=3) |
| NSRF-T w/o SGT [ours, this run] | 0.46/0.68 | 0.14/0.22 | 0.22/0.39 | 0.18/0.33 | 0.13/0.23 | 0.226±0.002/0.370±0.006 (n=3) |
| NSRF-T w/o MET [ours, this run] | 0.48/0.70 | 0.13/0.20 | 0.22/0.39 | 0.17/0.31 | 0.13/0.23 | 0.227±0.000/0.365±0.002 (n=3) |
| NSRF-T w/o NSRE [ours, this run] | 0.50/0.72 | 0.13/0.20 | 0.22/0.39 | 0.18/0.31 | 0.13/0.23 | 0.232±0.003/0.370±0.008 (n=3) |
| NSRF-T (full) [ours, this run] | 0.48/0.70 | 0.13/0.19 | 0.22/0.39 | 0.17/0.31 | 0.13/0.23 | 0.227±0.002/0.364±0.002 (n=3) |
| NSRF-T v2 w/o diffusion refinement [ours, this run] | 0.47/0.68 | 0.13/0.20 | 0.22/0.40 | 0.17/0.31 | 0.13/0.23 | 0.226±0.003/0.364±0.004 (n=3) |
| NSRF-T v2 (full) [ours, this run] | 0.48/0.72 | 0.14/0.21 | 0.22/0.39 | 0.17/0.30 | 0.13/0.23 | 0.226±0.001/0.371±0.006 (n=3) |

## ETH/UCY top-1, collisions and goal-conditioned realism (averaged over 5 scenes)
| Method | ADE₁ | FDE₁ | Collision rate, top-1 (%) | GC Fréchet (m) ↓ | GC path length (m) | GC tortuosity | real tortuosity |
|---|---|---|---|---|---|---|---|
| Constant velocity (det.) | 0.534 | 1.148 | 6.407 | 0.220 | 3.109 | 1.000 | 1.040 |
| Social Force Model | 0.712 | 1.339 | 0.008 | 0.364 | 3.328 | 1.039 | 1.040 |
| LSTM (K-mode) | 0.548 | 1.152 | 6.416 | 0.211 | 3.170 | 1.014 | 1.040 |
| Transformer (no RAMG/SGT/MET) | 0.542 | 1.141 | 6.320 | 0.208 | 3.166 | 1.016 | 1.040 |
| NSRF-T w/o RAMG | 0.535 | 1.112 | 1.903 | 0.215 | 3.183 | 1.024 | 1.040 |
| NSRF-T w/o SGT | 0.544 | 1.143 | 2.171 | 0.212 | 3.172 | 1.019 | 1.040 |
| NSRF-T w/o MET | 0.541 | 1.120 | 1.943 | 0.213 | 3.174 | 1.020 | 1.040 |
| NSRF-T w/o NSRE | 0.541 | 1.118 | 6.178 | 0.214 | 3.179 | 1.022 | 1.040 |
| NSRF-T (full) | 0.544 | 1.128 | 1.922 | 0.214 | 3.183 | 1.023 | 1.040 |
| NSRF-T v2 w/o diffusion refinement | 0.540 | 1.122 | 2.018 | 0.214 | 3.182 | 1.023 | 1.040 |
| NSRF-T v2 (full) | 0.544 | 1.139 | 1.990 | 0.213 | 3.175 | 1.020 | 1.040 |

## Hospital twin: realism, safety, efficiency
| Method | Human-likeness (HLS) ↑ | Group-fraction error vs real ↓ | Collision rate (normal) ↓ | Route length / geodesic | Tasks per agent-hour ↑ | Tasks per staff/patient-hour ↑ | ms / step |
|---|---|---|---|---|---|---|---|
| Social Force Model (n=10) | 93.2 ± 0.1 | 0.163 ± 0.002 | 0.0045 ± 0.0002 | 1.33 ± 0.00 | 21.8 ± 0.4 | 22.0 ± 0.4 | 40.8 ± 0.2 |
| A* path following (n=10) | 92.2 ± 0.2 | 0.166 ± 0.001 | 0.0241 ± 0.0014 | 1.43 ± 0.01 | 22.8 ± 0.3 | 22.9 ± 0.4 | 40.9 ± 0.3 |
| Generative-Agents-lite (n=10) | 93.4 ± 0.1 | 0.163 ± 0.002 | 0.0039 ± 0.0002 | 1.33 ± 0.00 | 22.9 ± 0.4 | 22.2 ± 0.5 | 47.0 ± 7.1 |
| Emotion-Based Planner (n=10) | 91.8 ± 0.1 | 0.169 ± 0.002 | 0.0232 ± 0.0010 | 1.44 ± 0.01 | 23.2 ± 0.3 | 23.2 ± 0.4 | 43.1 ± 1.5 |
| NeuroSocial-RAGFormer (n=10) | 92.0 ± 0.4 | 0.175 ± 0.003 | 0.0029 ± 0.0004 | 1.38 ± 0.01 | 14.7 ± 1.2 | 14.5 ± 1.3 | 48.0 ± 7.0 |

## Hospital twin: information diffusion and evacuation
| Method | Info T50 (s) ↓ | Info coverage ↑ | Evacuation T95 (s) ↓ | Mean evac. time (s) ↓ | Collision rate (evac.) ↓ | Peak exit density (p/m²) | East-exit share |
|---|---|---|---|---|---|---|---|
| Social Force Model (n=10) | 154 ± 22 | 1.00 ± 0.00 | 119 ± 1 | 85 ± 1 | 0.0032 ± 0.0004 | 0.56 ± 0.08 | 0.48 ± 0.02 |
| A* path following (n=10) | 157 ± 22 | 1.00 ± 0.00 | 110 ± 1 | 82 ± 1 | 0.0648 ± 0.0043 | 0.64 ± 0.14 | 0.47 ± 0.02 |
| Generative-Agents-lite (n=10) | 81 ± 15 | 1.00 ± 0.00 | 134 ± 3 | 91 ± 3 | 0.0049 ± 0.0005 | 0.60 ± 0.06 | 0.17 ± 0.01 |
| Emotion-Based Planner (n=10) | 99 ± 13 | 1.00 ± 0.00 | 110 ± 2 | 79 ± 2 | 0.0641 ± 0.0068 | 0.68 ± 0.10 | 0.48 ± 0.03 |
| NeuroSocial-RAGFormer (n=10) | 36 ± 25 | 0.99 ± 0.00 | 116 ± 4 | 86 ± 4 | 0.0030 ± 0.0005 | 0.75 ± 0.10 | 0.39 ± 0.04 |

## Hospital twin: social structure
| Method | Ties per agent | Contagion index |
|---|---|---|
| Social Force Model (n=10) | 0.00 ± 0.00 | n/a |
| A* path following (n=10) | 0.00 ± 0.00 | n/a |
| Generative-Agents-lite (n=10) | 0.36 ± 0.00 | n/a |
| Emotion-Based Planner (n=10) | 0.00 ± 0.00 | n/a |
| NeuroSocial-RAGFormer (n=10) | 1.52 ± 0.11 | 0.14 ± 0.09 |

## Ablation (realism)
| Method | Human-likeness (HLS) ↑ | Group-fraction error vs real ↓ | Collision rate (normal) ↓ | Route length / geodesic | Tasks per agent-hour ↑ | Tasks per staff/patient-hour ↑ | ms / step |
|---|---|---|---|---|---|---|---|
| NeuroSocial-RAGFormer (n=10) | 92.0 ± 0.4 | 0.175 ± 0.003 | 0.0029 ± 0.0004 | 1.38 ± 0.01 | 14.7 ± 1.2 | 14.5 ± 1.3 | 48.0 ± 7.0 |
| w/o RAMG (n=10) | 89.8 ± 0.2 | 0.180 ± 0.001 | 0.0020 ± 0.0001 | 1.40 ± 0.01 | 10.4 ± 0.5 | 10.0 ± 0.5 | 42.1 ± 0.8 |
| w/o MET (n=10) | 92.9 ± 0.2 | 0.167 ± 0.003 | 0.0025 ± 0.0002 | 1.39 ± 0.01 | 13.4 ± 0.2 | 13.4 ± 0.3 | 47.1 ± 5.3 |
| w/o SGT (n=10) | 91.9 ± 0.2 | 0.174 ± 0.002 | 0.0025 ± 0.0002 | 1.39 ± 0.01 | 15.2 ± 0.7 | 15.1 ± 0.7 | 45.6 ± 5.0 |
| w/o NSRE (n=10) | 93.1 ± 0.2 | 0.142 ± 0.003 | 0.0042 ± 0.0008 | 1.37 ± 0.01 | 13.9 ± 1.7 | 13.8 ± 1.7 | 47.9 ± 6.5 |
| w/o PPO (NSRE rules only) (n=10) | 93.7 ± 0.0 | 0.177 ± 0.001 | 0.0080 ± 0.0004 | 1.50 ± 0.01 | 19.8 ± 0.3 | 20.2 ± 0.4 | 49.6 ± 25.7 |
| v1 + task reward only (n=10) | 93.1 ± 0.4 | 0.171 ± 0.002 | 0.0030 ± 0.0003 | 1.41 ± 0.01 | 16.8 ± 1.2 | 16.7 ± 1.2 | 47.7 ± 9.6 |

## Ablation (events)
| Method | Info T50 (s) ↓ | Info coverage ↑ | Evacuation T95 (s) ↓ | Mean evac. time (s) ↓ | Collision rate (evac.) ↓ | Peak exit density (p/m²) | East-exit share |
|---|---|---|---|---|---|---|---|
| NeuroSocial-RAGFormer (n=10) | 36 ± 25 | 0.99 ± 0.00 | 116 ± 4 | 86 ± 4 | 0.0030 ± 0.0005 | 0.75 ± 0.10 | 0.39 ± 0.04 |
| w/o RAMG (n=10) | 34 ± 16 | 0.98 ± 0.01 | 113 ± 1 | 87 ± 1 | 0.0019 ± 0.0003 | 0.91 ± 0.12 | 0.23 ± 0.02 |
| w/o MET (n=10) | 33 ± 5 | 0.99 ± 0.00 | 124 ± 2 | 93 ± 1 | 0.0017 ± 0.0003 | 0.83 ± 0.06 | 0.27 ± 0.02 |
| w/o SGT (n=10) | 25 ± 5 | 1.00 ± 0.00 | 113 ± 1 | 85 ± 1 | 0.0026 ± 0.0004 | 0.73 ± 0.08 | 0.37 ± 0.03 |
| w/o NSRE (n=10) | 37 ± 6 | 1.00 ± 0.00 | 92 ± 10 | 66 ± 13 | 0.0094 ± 0.0064 | 6.38 ± 1.60 | 0.07 ± 0.05 |
| w/o PPO (NSRE rules only) (n=10) | 196 ± 41 | 1.00 ± 0.00 | 114 ± 2 | 84 ± 1 | 0.0081 ± 0.0006 | 0.64 ± 0.07 | 0.46 ± 0.02 |
| v1 + task reward only (n=10) | 49 ± 8 | 1.00 ± 0.00 | 111 ± 3 | 83 ± 3 | 0.0032 ± 0.0007 | 0.68 ± 0.07 | 0.38 ± 0.05 |

## HLS components (100·(1−JSD) per kinematic feature)
| Method | speed | acc | turn | nn |
|---|---|---|---|---|
| Social Force Model | 92.7 ± 0.4 | 94.2 ± 0.3 | 88.8 ± 0.3 | 97.1 ± 0.1 |
| A* path following | 92.6 ± 0.3 | 90.5 ± 0.4 | 86.8 ± 0.3 | 98.7 ± 0.1 |
| Generative-Agents-lite | 92.5 ± 0.3 | 94.8 ± 0.2 | 89.9 ± 0.1 | 96.6 ± 0.2 |
| Emotion-Based Planner | 92.8 ± 0.3 | 89.4 ± 0.3 | 86.3 ± 0.2 | 98.8 ± 0.1 |
| NeuroSocial-RAGFormer | 90.2 ± 1.2 | 93.3 ± 0.3 | 87.7 ± 0.3 | 96.7 ± 0.4 |

Real ETH/UCY group fraction: 0.195

## Scalability (scalability_cuda.json)
| Method | N | ms/step | steps/s | real-time factor | peak memory (GB) | device |
|---|---|---|---|---|---|---|
| Social Force Model | 100 | 4.2 | 237.0 | 94.8× | 0.00 | cuda |
| Social Force Model | 500 | 4.7 | 213.6 | 85.4× | 0.02 | cuda |
| Social Force Model | 1000 | 4.7 | 210.5 | 84.2× | 0.07 | cuda |
| Social Force Model | 2000 | 5.0 | 199.6 | 79.9× | 0.25 | cuda |
| Social Force Model | 5000 | 10.9 | 91.5 | 36.6× | 1.49 | cuda |
| A* path following | 100 | 4.0 | 250.6 | 100.3× | 0.00 | cuda |
| A* path following | 500 | 4.6 | 218.2 | 87.3× | 0.02 | cuda |
| A* path following | 1000 | 4.8 | 208.6 | 83.4× | 0.06 | cuda |
| A* path following | 2000 | 4.9 | 202.3 | 80.9× | 0.22 | cuda |
| A* path following | 5000 | 9.6 | 103.8 | 41.5× | 1.30 | cuda |
| Generative-Agents-lite | 100 | 4.5 | 224.6 | 89.8× | 0.00 | cuda |
| Generative-Agents-lite | 500 | 5.8 | 171.5 | 68.6× | 0.02 | cuda |
| Generative-Agents-lite | 1000 | 6.3 | 159.4 | 63.8× | 0.07 | cuda |
| Generative-Agents-lite | 2000 | 6.6 | 150.6 | 60.2× | 0.25 | cuda |
| Generative-Agents-lite | 5000 | 12.7 | 79.0 | 31.6× | 1.49 | cuda |
| Emotion-Based Planner | 100 | 4.5 | 222.4 | 89.0× | 0.00 | cuda |
| Emotion-Based Planner | 500 | 5.3 | 187.3 | 74.9× | 0.02 | cuda |
| Emotion-Based Planner | 1000 | 5.5 | 180.6 | 72.2× | 0.06 | cuda |
| Emotion-Based Planner | 2000 | 5.7 | 175.8 | 70.3× | 0.22 | cuda |
| Emotion-Based Planner | 5000 | 10.5 | 95.5 | 38.2× | 1.30 | cuda |
| v1 + task reward only | 100 | 5.4 | 183.6 | 73.4× | 0.04 | cuda |
| v1 + task reward only | 500 | 6.6 | 151.0 | 60.4× | 0.06 | cuda |
| v1 + task reward only | 1000 | 6.8 | 147.0 | 58.8× | 0.10 | cuda |
| v1 + task reward only | 2000 | 7.3 | 137.0 | 54.8× | 0.28 | cuda |
| v1 + task reward only | 5000 | 15.8 | 63.1 | 25.3× | 1.53 | cuda |

## Multi-day continuity (Scenario 5)
| Run | Day | Top-5 ties | RSI vs previous day | Memories carried from earlier days | Slots used |
|---|---|---|---|---|---|
| ga_s0 | 1 | 0.36 | – | 0.0 | 14.4 |
| ga_s0 | 2 | 0.36 | 1.000 | 13.3 | 27.3 |
| ga_s0 | 3 | 0.36 | 1.000 | 17.3 | 31.4 |
| ga_s0 | 4 | 0.36 | 1.000 | 18.3 | 31.9 |
| ga_s0 | 5 | 0.36 | 1.000 | 18.2 | 32.0 |
| ga_s0 | 6 | 0.36 | 1.000 | 18.9 | 32.0 |
| ga_s0 | 7 | 0.36 | 1.000 | 19.0 | 32.0 |
| ga_s1 | 1 | 0.36 | – | 0.0 | 13.9 |
| ga_s1 | 2 | 0.36 | 1.000 | 12.9 | 27.2 |
| ga_s1 | 3 | 0.36 | 1.000 | 17.5 | 31.4 |
| ga_s1 | 4 | 0.36 | 1.000 | 17.9 | 32.0 |
| ga_s1 | 5 | 0.36 | 1.000 | 18.7 | 32.0 |
| ga_s1 | 6 | 0.36 | 1.000 | 18.6 | 32.0 |
| ga_s1 | 7 | 0.36 | 1.000 | 18.5 | 32.0 |
| ga_s2 | 1 | 0.36 | – | 0.0 | 14.2 |
| ga_s2 | 2 | 0.36 | 1.000 | 13.2 | 27.2 |
| ga_s2 | 3 | 0.36 | 1.000 | 17.2 | 31.3 |
| ga_s2 | 4 | 0.36 | 1.000 | 18.4 | 31.9 |
| ga_s2 | 5 | 0.36 | 1.000 | 18.6 | 32.0 |
| ga_s2 | 6 | 0.36 | 1.000 | 18.3 | 32.0 |
| ga_s2 | 7 | 0.36 | 1.000 | 18.4 | 32.0 |
| nsrf_v2_task_only_s0 | 1 | 1.95 | – | 0.0 | 23.2 |
| nsrf_v2_task_only_s0 | 2 | 2.51 | 0.452 | 9.7 | 30.0 |
| nsrf_v2_task_only_s0 | 3 | 3.09 | 0.455 | 10.2 | 31.3 |
| nsrf_v2_task_only_s0 | 4 | 3.35 | 0.478 | 10.8 | 31.9 |
| nsrf_v2_task_only_s0 | 5 | 3.38 | 0.497 | 9.6 | 32.0 |
| nsrf_v2_task_only_s0 | 6 | 3.51 | 0.489 | 9.5 | 32.0 |
| nsrf_v2_task_only_s0 | 7 | 3.49 | 0.478 | 9.0 | 32.0 |
| nsrf_v2_task_only_s1 | 1 | 1.89 | – | 0.0 | 21.5 |
| nsrf_v2_task_only_s1 | 2 | 2.42 | 0.489 | 7.7 | 29.8 |
| nsrf_v2_task_only_s1 | 3 | 2.95 | 0.467 | 10.4 | 31.1 |
| nsrf_v2_task_only_s1 | 4 | 3.07 | 0.497 | 10.5 | 31.8 |
| nsrf_v2_task_only_s1 | 5 | 3.31 | 0.524 | 10.1 | 32.0 |
| nsrf_v2_task_only_s1 | 6 | 3.31 | 0.519 | 10.0 | 32.0 |
| nsrf_v2_task_only_s1 | 7 | 3.24 | 0.488 | 9.3 | 32.0 |
| nsrf_v2_task_only_s2 | 1 | 1.95 | – | 0.0 | 21.5 |
| nsrf_v2_task_only_s2 | 2 | 2.54 | 0.450 | 8.6 | 30.1 |
| nsrf_v2_task_only_s2 | 3 | 3.09 | 0.472 | 9.4 | 31.2 |
| nsrf_v2_task_only_s2 | 4 | 3.28 | 0.470 | 10.4 | 31.8 |
| nsrf_v2_task_only_s2 | 5 | 3.28 | 0.525 | 10.3 | 32.0 |
| nsrf_v2_task_only_s2 | 6 | 3.36 | 0.484 | 9.4 | 32.0 |
| nsrf_v2_task_only_s2 | 7 | 3.19 | 0.514 | 9.3 | 32.0 |