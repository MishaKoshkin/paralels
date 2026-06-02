import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Функция для отрисовки таблицы как отдельного изображения
# ------------------------------------------------------------
def draw_table_image(pivot_df, output_filename, title=None):
    """
    Создаёт PNG-изображение, содержащее только таблицу с голубым фоном.
    pivot_df: pandas DataFrame со строками = потоки, столбцами = размеры
    output_filename: имя выходного файла
    title: опциональный заголовок над таблицей
    """
    # Получаем данные из DataFrame
    headers = list(pivot_df.columns)           # размеры
    rows = pivot_df.values.tolist()            # ускорения (числа или прочерки)
    row_labels = pivot_df.index.tolist()       # количество потоков

    # Формируем полную таблицу с первым столбцом "Потоки"
    col_labels = ["Потоки"] + headers
    cell_text = []
    for i, thr in enumerate(row_labels):
        row = [thr] + rows[i]
        # Преобразуем числа в строки с округлением (уже округлено в pivot)
        row = [str(x) if x != '-' else '-' for x in row]
        cell_text.append(row)

    # Определяем размеры фигуры
    n_rows = len(cell_text) + 1  # +1 для заголовка
    n_cols = len(col_labels)
    fig_width = max(8, n_cols * 1.2)
    fig_height = max(3, n_rows * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Цвета: тёмно-синий для заголовков, голубой для данных
    header_colors = ['#1f4e79'] * n_cols
    data_colors = ['#87CEEB'] * n_cols

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=header_colors)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Оформление ячеек
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # строка заголовка
            cell.set_facecolor(header_colors[col])
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor(data_colors[col % len(data_colors)])
            cell.set_text_props(color='black')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Таблица сохранена как '{output_filename}'")

# ------------------------------------------------------------
# Чтение данных из файла vector-matrix_benchmark.txt
# ------------------------------------------------------------
filename = "build/benchmark_data.txt"
data = []

if not os.path.exists(filename):
    print(f"Файл {filename} не найден. Проверьте путь.")
    exit(1)

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            size = int(parts[0])
            threads = int(parts[1])
            time = float(parts[2])
            speedup = float(parts[3])
            data.append([size, threads, time, speedup])
        except ValueError:
            continue

if not data:
    print("Файл не содержит корректных данных.")
    exit(1)

# Создаём DataFrame
df = pd.DataFrame(data, columns=['size', 'threads', 'time', 'speedup'])

# ------------------------------------------------------------
# 1. Таблица (сводная): строки = потоки, столбцы = размеры
# ------------------------------------------------------------
print("\n=== Ускорение matrix-vector умножения ===\n")
pivot = df.pivot(index='threads', columns='size', values='speedup')
pivot = pivot.sort_index()
pivot = pivot.round(3)
pivot = pivot.fillna('-')
print(pivot.to_string())
print("\n")

# ------------------------------------------------------------
# 2. График ускорения от числа потоков
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

sizes = df['size'].unique()
for size in sorted(sizes):
    sub = df[df['size'] == size].sort_values('threads')
    plt.plot(sub['threads'], sub['speedup'], 'o-', linewidth=2, markersize=6, label=f'Размер {size}')

max_threads = df['threads'].max()
plt.plot([1, max_threads], [1, max_threads], 'k--', linewidth=1.5, label='Идеальное ускорение')

plt.xlabel('Количество потоков', fontsize=12)
plt.ylabel('Ускорение', fontsize=12)
plt.title('Matrix-Vector умножение', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('vector-matrix_speedup.png', dpi=150)
print("График сохранён как 'vector-matrix_speedup.png'")

# ------------------------------------------------------------
# 3. Отдельная картинка с таблицей
# ------------------------------------------------------------
draw_table_image(pivot, 'vector-matrix_table.png', title='Matrix-Vector умножение (ускорения)')

plt.show()

# ------------------------------------------------------------
# 4. График прибыли (прибыль = ускорение - 0.01*(потоки-1)^2)
# ------------------------------------------------------------
df['profit'] = df['speedup'] - 0.01 * (df['threads'] - 1)**2

plt.figure(figsize=(8, 6))

for size in sorted(sizes):
    sub = df[df['size'] == size].sort_values('threads')
    plt.plot(sub['threads'], sub['profit'], 's-', linewidth=2, markersize=6, label=f'Размер {size}')

# Горизонтальная линия на уровне 0 для наглядности
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

plt.xlabel('Количество потоков', fontsize=12)
plt.ylabel('Прибыль', fontsize=12)
plt.title('Прибыль = ускорение - 0.01*(потоки-1)²', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('vector-matrix_profit.png', dpi=150)
print("График прибыли сохранён как 'vector-matrix_profit.png'")