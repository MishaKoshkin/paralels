import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# 1. Чтение и построение для vector-matrix (4 столбца: размер, потоки, время, ускорение)
# ------------------------------------------------------------
def plot_vector_matrix(filename, output_image):
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            size = int(parts[0])
            threads = int(parts[1])
            speedup = float(parts[3])
            if size not in data:
                data[size] = {'threads': [], 'speedup': []}
            data[size]['threads'].append(threads)
            data[size]['speedup'].append(speedup)
    
    if not data:
        print(f"Нет данных в {filename}")
        return
    
    plt.figure(figsize=(8, 6))
    for size, vals in data.items():
        # Сортировка по числу потоков
        thr = np.array(vals['threads'])
        sp = np.array(vals['speedup'])
        idx = np.argsort(thr)
        thr = thr[idx]
        sp = sp[idx]
        plt.plot(thr, sp, 'o-', linewidth=2, markersize=6, label=f'Размер {size}')
    
    # Идеальное ускорение
    max_threads = max(max(vals['threads']) for vals in data.values())
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='Идеальное ускорение')
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Ускорение', fontsize=12)
    plt.title('Matrix-Vector умножение', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Сохранён {output_image}")

# ------------------------------------------------------------
# 2. Чтение для integrate (3 столбца: потоки, время, ускорение)
# ------------------------------------------------------------
def plot_integrate(filename, output_image):
    threads = []
    speedup = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            thr = int(parts[0])
            sp = float(parts[2])
            threads.append(thr)
            speedup.append(sp)
    
    if not threads:
        print(f"Нет данных в {filename}")
        return
    
    # Сортировка
    idx = np.argsort(threads)
    thr = np.array(threads)[idx]
    sp = np.array(speedup)[idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(thr, sp, 'o-', color='blue', linewidth=2, markersize=6, label='Integrate')
    
    max_threads = max(thr)
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='Идеальное ускорение')
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Ускорение', fontsize=12)
    plt.title('Интегрирование (метод прямоугольников)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Сохранён {output_image}")

# ------------------------------------------------------------
# 3. Чтение для slau (6 столбцов: потоки, serial, split, single, speedup_split, speedup_single)
# ------------------------------------------------------------
def plot_slau(filename, output_image):
    threads = []
    speedup_split = []
    speedup_single = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            thr = int(parts[0])
            sp_split = float(parts[4])
            sp_single = float(parts[5])
            threads.append(thr)
            speedup_split.append(sp_split)
            speedup_single.append(sp_single)
    
    if not threads:
        print(f"Нет данных в {filename}")
        return
    
    idx = np.argsort(threads)
    thr = np.array(threads)[idx]
    sp_split = np.array(speedup_split)[idx]
    sp_single = np.array(speedup_single)[idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(thr, sp_split, 'o-', color='green', linewidth=2, markersize=6, label='SLAU (Split)')
    plt.plot(thr, sp_single, 's-', color='lime', linewidth=2, markersize=6, label='SLAU (Single region)')
    
    max_threads = max(thr)
    plt.plot([1, max_threads], [1, max_threads], 'k--', label='Идеальное ускорение')
    
    plt.xlabel('Количество потоков', fontsize=12)
    plt.ylabel('Ускорение', fontsize=12)
    plt.title('Решение СЛАУ (метод Ричардсона)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"Сохранён {output_image}")

# ------------------------------------------------------------
# 4. Обработка schedule_slau.txt: вывод таблицы и построение графика
# ------------------------------------------------------------
def process_schedule(filename, output_image=None):
    """
    Читает файл schedule_slau.txt формата:
    # комментарии
    schedule_type  time(s)  error
    Возвращает словарь {schedule: время} и выводит таблицу.
    Если задан output_image, строит столбчатую диаграмму.
    """
    schedules = []
    times = []
    errors = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            # Предполагаем: schedule_type время error
            sched = parts[0]
            # время может быть как 0.123456, так и в научной нотации
            try:
                t = float(parts[1])
                err = float(parts[2])
            except ValueError:
                continue
            schedules.append(sched)
            times.append(t)
            errors.append(err)
    
    if not schedules:
        print(f"Файл {filename} не содержит данных или имеет неверный формат.")
        return
    
    # Вывод таблицы в консоль
    print("\n" + "="*60)
    print("Результаты исследования schedule (время выполнения, с)")
    print("="*60)
    print(f"{'Schedule':<20} {'Time (s)':<12} {'Error':<12}")
    print("-"*60)
    for sched, t, err in zip(schedules, times, errors):
        # округление до тысячных
        print(f"{sched:<20} {t:>10.3f}   {err:>10.2e}")
    print("="*60)
    
    # Построение графика, если указан output_image
    if output_image:
        plt.figure(figsize=(10, 6))
        # Цвета для разных типов расписаний
        colors = plt.cm.tab20(np.linspace(0, 1, len(schedules)))
        bars = plt.bar(schedules, times, color=colors, edgecolor='black')
        plt.ylabel('Время выполнения (секунды)', fontsize=12)
        plt.xlabel('Тип расписания (schedule)', fontsize=12)
        plt.title('Сравнение времени решения СЛАУ для разных schedule', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        
        # Добавление значений на столбцы (с округлением до тысячных)
        for bar, t in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(times),
                     f'{t:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_image, dpi=150)
        plt.close()
        print(f"Сохранён график schedule: {output_image}")

# ------------------------------------------------------------
# Главная функция
# ------------------------------------------------------------
def main():
    # Имена файлов (при необходимости измените)
    files = {
        'build/vector-matrix_benchmark.txt': 'vector-matrix_speedup.png',
        'build/integrate_benchmark.txt': 'integrate_speedup.png',
        'build/slau-benchmark.txt': 'slau_speedup.png'
    }
    
    for infile, outfile in files.items():
        if not os.path.exists(infile):
            print(f"Файл {infile} не найден, пропускаем.")
            continue
        
        # Определяем формат по количеству столбцов в первой не-комментарной строке
        with open(infile, 'r') as f:
            first_data_line = None
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                first_data_line = line
                break
        if not first_data_line:
            print(f"Файл {infile} пуст, пропускаем.")
            continue
        
        cols = len(first_data_line.split())
        if cols == 4:
            plot_vector_matrix(infile, outfile)
        elif cols == 3:
            plot_integrate(infile, outfile)
        elif cols == 6:
            plot_slau(infile, outfile)
        else:
            print(f"Неизвестный формат файла {infile} (столбцов: {cols})")
    
    # Дополнительная обработка schedule_slau.txt, если файл существует
    schedule_file = 'build/schedule_slau.txt'   # можно указать свой путь
    schedule_image = 'schedule_comparison.png'
    if os.path.exists(schedule_file):
        process_schedule(schedule_file, schedule_image)
    else:
        # Попробуем в текущей директории
        schedule_file_alt = 'schedule_slau.txt'
        if os.path.exists(schedule_file_alt):
            process_schedule(schedule_file_alt, schedule_image)
        else:
            print(f"\nФайл {schedule_file} не найден. Обработка schedule пропущена.")

if __name__ == '__main__':
    main()