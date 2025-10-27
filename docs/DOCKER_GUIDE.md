# 🐳 Docker GPU Training Guide

## Quick Start on GPU Server

### 1. Подготовка
```bash
# Клонируем репозиторий
git clone https://github.com/nikitashvarts/scimdix_aspect_extraction.git
cd scimdix_aspect_extraction

# Делаем скрипты исполняемыми
chmod +x *.sh
```

### 2. Быстрый тест
```bash
# Проверяем что все работает (1 эпоха, малый batch)
./run_single_experiment.sh test
```

### 3. Запуск отдельных экспериментов

```bash
# Baseline эксперименты
./run_single_experiment.sh baseline_ru
./run_single_experiment.sh baseline_kz

# Основной zero-shot эксперимент
./run_single_experiment.sh zero_shot_ru_to_kz

# LODO эксперименты
./run_single_experiment.sh lodo_it
./run_single_experiment.sh lodo_ling
./run_single_experiment.sh lodo_med
./run_single_experiment.sh lodo_psy
```

### 4. Настройка параметров
```bash
# Кастомный batch size и количество эпох
./run_single_experiment.sh baseline_ru 16 10
./run_single_experiment.sh zero_shot_ru_to_kz 32 25
```

### 5. Запуск всех экспериментов сразу
```bash
# ВНИМАНИЕ: займет много времени!
./run_all_experiments.sh
```

## 📊 Результаты

Все результаты сохраняются в папку `results/`:
- `results/experiments/<experiment_name>/results.json` - метрики
- `results/models/<experiment_name>/` - сохраненные модели
- `results/logs/<experiment_name>/` - логи обучения

## ⚙️ Настройки GPU

Скрипты настроены для использования **GPU=1** (вторая видеокарта).
Если нужно изменить:

```bash
# Редактируем переменную в скриптах
export CUDA_VISIBLE_DEVICES=0  # для первой GPU
# или
export CUDA_VISIBLE_DEVICES=1  # для второй GPU
```

## 🐛 Troubleshooting

### Проверка GPU
```bash
# Проверяем доступность GPU в Docker
docker run --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### Логи обучения
```bash
# Смотрим логи последнего контейнера
docker logs $(docker ps -lq)
```

### Очистка Docker
```bash
# Удаляем старые контейнеры и образы
docker system prune -f
```

## 📈 Ожидаемое время выполнения

На RTX 3090:
- **test**: ~5-10 минут
- **baseline_ru/kz**: ~2-3 часа
- **zero_shot_ru_to_kz**: ~2-3 часа  
- **lodo_***: ~1-2 часа каждый
- **Все эксперименты**: ~15-20 часов