#  Код модуля классификации, использующийся для фильтрации боксов на выходе CenterPoint 

Для того, чтобы запустить классификатор на тестовых данных, выполните: 
```
python test_mlp.py
```
mlp_4cls_3_layers_slice_3_hm.pt - файл весов для классификатора
prediction.pkl - файл выходного словаря CenterPoint
sep_head0.pt - файл промежуточного текнзора CetnerPoint
