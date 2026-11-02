# TRIPOD+AI checklist - pointers (fill page numbers in the manuscript)
1 Title: screening (not diagnosis) model, children 18-72 months, Bangladesh
3 Background / 4 Objectives: screening vs M-CHAT-R/F; decision support only
5 Data sources: prospective protocol recordings, sites, dates  (NB00 audit_report.json)
6 Participants: eligibility, ASD / TD / DD groups, consent
7 Data preparation: NB01 / NB02 pipelines, missing-modality handling
8 Outcome: clinician DSM-5 (+ADOS-2) diagnosis blind to the model
9 Predictors: face video, voice, cry, M-CHAT items (reported with/without)
10 Sample size: statistician calculation (attach)
11 Missing data: learned missing tokens + modality dropout
12 Analysis: child-wise CV, locked test (NB00), threshold at 85 % sensitivity on OOF
13 Class imbalance / 14 Fairness: table_fairness.csv
15 Model output: risk + uncertainty, 3-way decision (table_triage.csv)
16 Training vs evaluation: table_main_metrics_ci.csv (OOF vs locked test)
20 Performance: AUROC/AUPRC/sens/spec/PPV/NPV/calibration with 95 % CIs
22 Model updating / 23 interpretability: NB10 outputs
25 Shortcut / confounding analysis: table_ablations_shortcuts.csv
27 Code & data availability: Kaggle notebooks + asd_lib; data per consent
