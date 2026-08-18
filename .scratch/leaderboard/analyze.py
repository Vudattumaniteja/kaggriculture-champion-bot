import pandas as pd
import numpy as np

csv_path = r'C:\Users\Manit\Desktop\kaggle\.scratch\leaderboard\kaggriculture-publicleaderboard-2026-08-17T12_08_50.csv'
df = pd.read_csv(csv_path)

stats = {
    'total_teams': len(df),
    'total_submissions': int(df['SubmissionCount'].sum()),
    'mean_score': float(df['Score'].mean()),
    'median_score': float(df['Score'].median()),
    'p75_score': float(df['Score'].quantile(0.75)),
    'p90_score': float(df['Score'].quantile(0.90)),
    'p95_score': float(df['Score'].quantile(0.95)),
    'p99_score': float(df['Score'].quantile(0.99)),
    'max_score': float(df['Score'].max()),
    'min_score': float(df['Score'].min()),
}

top_teams = df.sort_values(by='Rank', ascending=True).head(15)

report = ['# Public Leaderboard Intelligence Report', '']
report.append('**Snapshot Timestamp:** 2026-08-17 12:08:50 UTC')
report.append(f"**Total Competing Teams:** {stats['total_teams']:,}")
report.append(f"**Total Submissions:** {stats['total_submissions']:,}")
report.append('')
report.append('## 1. Score Distribution & Rating Tiers')
report.append('')
report.append('| Metric / Percentile | Score (Elo / Rating) | Meaning / Tier |')
report.append('| :--- | :--- | :--- |')
report.append(f"| **Maximum (Rank #1)** | **{stats['max_score']:.1f}** | Top Grandmaster Elite Tier |")
report.append(f"| **99th Percentile** | **{stats['p99_score']:.1f}** | Top 1% (Gold Medal Zone) |")
report.append(f"| **95th Percentile** | **{stats['p95_score']:.1f}** | Top 5% (Silver Medal Zone) |")
report.append(f"| **90th Percentile** | **{stats['p90_score']:.1f}** | Top 10% (Bronze Medal Zone) |")
report.append(f"| **75th Percentile** | **{stats['p75_score']:.1f}** | Above Average Baseline |")
report.append(f"| **Median (50th %)** | **{stats['median_score']:.1f}** | Default Baseline / Average |")
report.append(f"| **Mean Score** | **{stats['mean_score']:.1f}** | Population Mean |")
report.append(f"| **Minimum Score** | **{stats['min_score']:.1f}** | Inactive / Crashing / Bankrupt Bots |")
report.append('')
report.append('## 2. Top 15 Leaderboard Standings')
report.append('')
report.append('| Rank | Team Name | Rating Score | Submissions | Member(s) |')
report.append('| :---: | :--- | :---: | :---: | :--- |')

for _, row in top_teams.iterrows():
    name = str(row['TeamName']).encode('ascii', 'ignore').decode('ascii')
    members = str(row['TeamMemberUserNames']).encode('ascii', 'ignore').decode('ascii')
    report.append(f"| **#{int(row['Rank'])}** | {name} | **{float(row['Score']):.1f}** | {int(row['SubmissionCount'])} | `{members}` |")

with open(r'C:\Users\Manit\Desktop\kaggle\.scratch\leaderboard\LEADERBOARD_ANALYSIS.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print('Report generated successfully!')
