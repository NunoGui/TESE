import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
 
# ── Dados ──────────────────────────────────────────────────────────────────
metricas = ['Recall@10', 'MRR@10', 'NDCG@10', 'Hit@10', 'Precision@10']
lightgcn = [0.7124,      0.4115,   0.4836,    0.7124,   0.0712]
ngcf     = [0.7124,      0.4116,   0.4837,    0.7124,   0.0712]
 
x = np.arange(len(metricas))
largura = 0.35
 
# ── Estilo ─────────────────────────────────────────────────────────────────
COR_LIGHTGCN = '#2E4057'   # azul escuro
COR_NGCF     = '#6B8F71'   # verde
COR_FUNDO    = '#F7F7F7'
COR_GRID     = '#E0E0E0'
 
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(COR_FUNDO)
ax.set_facecolor(COR_FUNDO)
 
# ── Barras ─────────────────────────────────────────────────────────────────
barras_lgcn = ax.bar(x - largura/2, lightgcn, largura,
                     label='LightGCN', color=COR_LIGHTGCN,
                     edgecolor='white', linewidth=0.8, zorder=3)
 
barras_ngcf = ax.bar(x + largura/2, ngcf, largura,
                     label='NGCF', color=COR_NGCF,
                     edgecolor='white', linewidth=0.8, zorder=3)
 
# ── Valores em cima das barras ─────────────────────────────────────────────
def adicionar_valores(barras, cor):
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(f'{altura:.4f}',
                    xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8.5, color=cor, fontweight='bold')
 
adicionar_valores(barras_lgcn, COR_LIGHTGCN)
adicionar_valores(barras_ngcf, COR_NGCF)
 
# ── Eixos e grelha ─────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=11, fontweight='bold', color='#333333')
ax.set_ylim(0, 0.90)
ax.set_ylabel('Valor (escala 0–1)', fontsize=11, color='#555555', labelpad=10)
ax.yaxis.set_tick_params(labelcolor='#777777', labelsize=9)
ax.grid(axis='y', color=COR_GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
 
for spine in ax.spines.values():
    spine.set_visible(False)
 
# ── Título e legenda ───────────────────────────────────────────────────────
ax.set_title('Comparação LightGCN vs NGCF — EmoRecSys',
             fontsize=14, fontweight='bold', color='#222222', pad=20)
 
subtitulo = 'Dataset: EmoRecSys  |  Split: 80/10/10  |  Avaliação: Top-10  |  Biblioteca: RecBole'
fig.text(0.5, 0.01, subtitulo, ha='center', fontsize=8.5, color='#999999')
 
legenda = [mpatches.Patch(color=COR_LIGHTGCN, label='LightGCN'),
           mpatches.Patch(color=COR_NGCF,     label='NGCF')]
ax.legend(handles=legenda, loc='upper right', fontsize=10,
          framealpha=0.9, edgecolor=COR_GRID)
 
# ── Guardar e mostrar ──────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('grafico_comparativo.png', dpi=180, bbox_inches='tight',
            facecolor=COR_FUNDO)
plt.show()
 
print("Gráfico guardado como 'grafico_comparativo.png'")
 