# XMF-Photo
Unified 7-class facial-expression dataset (neutral, happy, sad, surprise, fear, disgust, anger), 50109 images at 160 px RGB.
Sources: FER2013 (msambare/fer2013), RAF-DB (shuvoalok/raf-db-dataset), CK+ (davilsena/ckdataset; external test only).
Splits: official test sets kept; 10 % of training as validation; images with identical pHash removed (test > val > train priority).
Use: pre-training of the face-expression encoder. Not an autism dataset. Licences: as stated by each original source (research use).

| source   |   test |   train |   val |
|:---------|-------:|--------:|------:|
| ckplus   |    867 |       0 |     0 |
| fer2013  |   7080 |   24124 |  2707 |
| rafdb    |   3067 |   10971 |  1293 |
