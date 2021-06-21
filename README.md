# Точка схода.

Для тестирования модели необходимо выполнить следующие команды:

1) !python test_gen.py --s path_to_dataset --d path_to_save_new_dataset --num number_of_images --seed random_seed - генерация данных для теста
2) !python make_pred.py --model path_to_model(default: './model.pt') --test path_to_test_dataset --pred path_to_json_to_save_predictions(default:'./predictions.json') --gt path_to_gt_json - сохранение json-файла с предсказаниями
3) !python metrics.py --imgs path_to_test_dataset --gt path_to_gt_json --ans path_to_json_with_predictions - подсчет метрики

Перед запуском нужно сохранить веса модели https://drive.google.com/file/d/1Y9np4lTNMUGdNinnkjLJnonFq5YgQeu1/view?usp=sharing и поместить ее в общую папку. В общем фолдере должны быть: все скрипты; папка с исходными данными source; ground truth json - markup.json; пустой файл для предсказаний - predictions.json; пустая папка test для новых данных, в ней пустой файл markup.json.
