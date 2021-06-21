# Точка схода.

Для тестирования модели необходимо выполнить следующее:

Скачать и распаковать архив **VP-main** https://drive.google.com/file/d/17KruSAywPNDZ2HmaEMTzSIduPWPb5QMx/view?usp=sharing и, находясь в этой директории, запустить:
1) **python test_gen.py --s ./ --d ./test/ --num 50 --seed 0**
2) **python make_pred.py --model ./model.pt --test ./test/ --pred ./predictions.json --gt ./test/markup.json**
3) **python metrics.py --imgs ./test/ --gt ./test/markup.json --ans ./predictions.json**

Пример запуска в файле **vp_test.ipynb**.

**ЛИБО**

Перед запуском нужно сохранить веса модели https://drive.google.com/file/d/1Y9np4lTNMUGdNinnkjLJnonFq5YgQeu1/view?usp=sharing и поместить ее в общую папку. В общем фолдере должны быть: все скрипты; папка с исходными данными source; ground truth json - markup.json; пустая папка test для новых данных, в ней пустой файл markup.json. Выполнить команды:

1) **python test_gen.py --s path_to_dataset --d path_to_save_new_dataset --num number_of_images --seed random_seed** - генерация данных для теста
2) **python make_pred.py --model path_to_model --test path_to_test_dataset --pred path_to_json_to_save_predictions --gt path_to_gt_json - сохранение json**-файла с предсказаниями
3) **python metrics.py --imgs path_to_test_dataset --gt path_to_gt_json --ans path_to_json_with_predictions** - подсчет метрики






