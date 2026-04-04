import numpy as np
import matplotlib
matplotlib.use('Agg')   # save to file instead of opening a window
import matplotlib.pyplot as plt

# ── load training logs ───────────────────────────────────────────────────────
baseline = np.load("baseline_rewards.npy")
marl     = np.load("marl_rewards.npy")
llm      = np.load("llm_rewards.npy")

baseline_keys = np.load("baseline_keys.npy")
marl_keys     = np.load("marl_keys.npy")
llm_keys      = np.load("llm_keys.npy")

# ── rolling average (100-episode window) ─────────────────────────────────────
def rolling_mean(data, window=100):
    result = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result[i] = np.mean(data[start:i+1])
    return result

episodes     = np.arange(len(baseline))
baseline_avg = rolling_mean(baseline)
marl_avg     = rolling_mean(marl)
llm_avg      = rolling_mean(llm)

baseline_keys_avg = rolling_mean(baseline_keys) * 100
marl_keys_avg     = rolling_mean(marl_keys) * 100
llm_keys_avg      = rolling_mean(llm_keys) * 100

# ── plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # 2 rows, 1 column, taller figure

# ==========================================
# TOP GRAPH: REWARDS
# ==========================================

# faint raw episode rewards in background for LLM
ax1.plot(episodes, llm, color='#3B82F6', alpha=0.10, linewidth=0.6, label='LLM individual episode rewards')

# rolling averages
ax1.plot(episodes, baseline_avg, label='Baseline (no communication)', color='#888888', linewidth=2.5)
ax1.plot(episodes, marl_avg,     label='MARL (emergent communication)', color='#E8A045', linewidth=2.5)
ax1.plot(episodes, llm_avg,      label='LLM sky agent (Gemini 2.5 Flash Lite)', color='#3B82F6', linewidth=2.5)

# y=0 reference
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.4, label='Zero reward threshold')

# labels and formatting
ax1.set_xlabel('Episode', fontsize=13)
ax1.set_ylabel('Average Reward (100-episode rolling mean)', fontsize=13)
ax1.set_title('Effect of Communication Protocol on Ground Agent Performance\n'
              '6×6 GridWorld — Key Collection + Exit Navigation Task\n'
              '(Task Reward)',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.set_xlim(0, len(episodes) - 1)
ax1.set_ylim(-15, 2)

# sort conditions down the right edge
conditions_reward = [
    ('Baseline', baseline_avg, '#888888'),
    ('MARL',     marl_avg,     '#E8A045'),
    ('LLM',      llm_avg,      '#3B82F6'),
]
y_offsets = [14, 0, -14]
for (label, data, color), y_off in zip(sorted(conditions_reward, key=lambda x: x[1][-1], reverse=True), y_offsets):
    ax1.annotate(f'{label}: {data[-1]:.2f}',
                 xy=(len(episodes) - 1, data[-1]),
                 xytext=(8, y_off), textcoords='offset points',
                 color=color, fontsize=10, fontweight='bold', va='center')


# ==========================================
# BOTTOM GRAPH: KEY COLLECTION
# ==========================================

# marking episodes where llm successfully got key
ax2.bar(episodes, llm_keys * 100, width=1.0, color='#3B82F6', alpha=0.10, label='LLM individual episode key success')

# rolling averages
ax2.plot(episodes, baseline_keys_avg, label='Baseline (no communication)', color='#888888', linewidth=2.5)
ax2.plot(episodes, marl_keys_avg,     label='MARL (emergent communication)', color='#E8A045', linewidth=2.5)
ax2.plot(episodes, llm_keys_avg,      label='LLM sky agent (Gemini 2.5 Flash Lite)', color='#3B82F6', linewidth=2.5)

# labels and formatting
ax2.set_xlabel('Episode', fontsize=13)
ax2.set_ylabel('Key Collection Rate % (100-episode rolling mean)', fontsize=13)
ax2.set_title('Key Collection Success Rate', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.25, linestyle='--')
ax2.set_xlim(0, len(episodes) - 1)
ax2.set_ylim(0, 105)

# sort conditions down the right edge
conditions_keys = [
    ('Baseline', baseline_keys_avg, '#888888'),
    ('MARL',     marl_keys_avg,     '#E8A045'),
    ('LLM',      llm_keys_avg,      '#3B82F6'),
]
for (label, data, color), y_off in zip(sorted(conditions_keys, key=lambda x: x[1][-1], reverse=True), y_offsets):
    ax2.annotate(f'{label}: {data[-1]:.1f}%',
                 xy=(len(episodes) - 1, data[-1]),
                 xytext=(8, y_off), textcoords='offset points',
                 color=color, fontsize=10, fontweight='bold', va='center')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("Plot saved to results.png — open it with: open results.png")
