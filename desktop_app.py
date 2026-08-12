import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

from model_core import (
    all_teams, build_match_explanation, data_diagnostics, find_upcoming_fixtures, load_csvs,
    predict_fixture, recent_matches_table, team_summary
)

class SoccerPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Soccer Predictor — Multi-Model 1X2')
        self.geometry('1180x780')
        self.minsize(980, 680)
        self.df = None
        self.result = None
        self.training = None
        self.files = []
        self.fixture_map = {}
        self._build()

    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')
        ttk.Label(top, text='Soccer Predictor', font=('Segoe UI', 20, 'bold')).grid(row=0, column=0, sticky='w')
        ttk.Label(top, text='Upload CSV history → select Home vs Away → run multi-model 1X2 analysis').grid(row=1, column=0, sticky='w', pady=(2,8))

        controls = ttk.LabelFrame(top, text='1. Data and match selection', padding=10)
        controls.grid(row=2, column=0, sticky='ew')
        top.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, weight=1)

        ttk.Button(controls, text='Upload CSV file(s)', command=self.load_files).grid(row=0, column=0, padx=(0,8), pady=4, sticky='w')
        self.file_label = ttk.Label(controls, text='No CSV files loaded')
        self.file_label.grid(row=0, column=1, columnspan=3, sticky='w')

        ttk.Label(controls, text='Upcoming fixture (optional)').grid(row=1, column=0, sticky='w', pady=4)
        self.fixture_var = tk.StringVar()
        self.fixture_box = ttk.Combobox(controls, textvariable=self.fixture_var, state='readonly')
        self.fixture_box.grid(row=1, column=1, columnspan=3, sticky='ew', pady=4)
        self.fixture_box.bind('<<ComboboxSelected>>', self.fixture_selected)

        ttk.Label(controls, text='Home team').grid(row=2, column=0, sticky='w', pady=4)
        self.home_var = tk.StringVar()
        self.home_box = ttk.Combobox(controls, textvariable=self.home_var, state='readonly')
        self.home_box.grid(row=2, column=1, sticky='ew', padx=(0,14), pady=4)

        ttk.Label(controls, text='Away team').grid(row=2, column=2, sticky='w', pady=4)
        self.away_var = tk.StringVar()
        self.away_box = ttk.Combobox(controls, textvariable=self.away_var, state='readonly')
        self.away_box.grid(row=2, column=3, sticky='ew', pady=4)

        ttk.Button(controls, text='ANALYSE MATCH', command=self.predict).grid(row=3, column=0, pady=(10,2), sticky='w')
        self.data_label = ttk.Label(controls, text='')
        self.data_label.grid(row=3, column=1, columnspan=3, sticky='w')

        self.tabs = ttk.Notebook(self, padding=6)
        self.tabs.pack(fill='both', expand=True, padx=10, pady=(0,10))
        self.pred_text = self._make_tab('Prediction')
        self.explain_text = self._make_tab('Explanation')
        self.str_text = self._make_tab('Strength / Weakness')
        self.form_text = self._make_tab('Recent Form')
        self.model_text = self._make_tab('Model Breakdown')
        self.method_text = self._make_tab('Method')
        self.method_text.insert('1.0', self._method_text())

    def _make_tab(self, title):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text=title)
        txt = tk.Text(frame, wrap='none', font=('Consolas', 10), padx=10, pady=10)
        y = ttk.Scrollbar(frame, orient='vertical', command=txt.yview)
        x = ttk.Scrollbar(frame, orient='horizontal', command=txt.xview)
        txt.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        txt.grid(row=0, column=0, sticky='nsew')
        y.grid(row=0, column=1, sticky='ns')
        x.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return txt

    def load_files(self):
        paths = filedialog.askopenfilenames(title='Select soccer CSV files', filetypes=[('CSV files','*.csv'),('All files','*.*')])
        if not paths:
            return
        try:
            self.df = load_csvs(list(paths))
        except Exception as e:
            messagebox.showerror('CSV parsing error', str(e))
            return
        self.files = list(paths)
        teams = all_teams(self.df)
        self.home_box['values'] = teams
        self.away_box['values'] = teams
        if teams:
            self.home_var.set(teams[0])
            self.away_var.set(teams[1] if len(teams) > 1 else teams[0])

        fixtures = find_upcoming_fixtures(self.df)
        labels = ['Manual team selection']
        self.fixture_map = {}
        for i, r in fixtures.iterrows():
            label = f"{r['date_GMT']} | {r['home_team_name']} vs {r['away_team_name']}"
            labels.append(label)
            self.fixture_map[label] = (r['home_team_name'], r['away_team_name'], float(r['timestamp']))
        self.fixture_box['values'] = labels
        self.fixture_var.set(labels[0])

        d = data_diagnostics(self.df)
        self.file_label.config(text=f"{len(paths)} file(s): " + ', '.join(os.path.basename(p) for p in paths))
        self.data_label.config(text=f"Parsed {d['rows']} rows | {d['completed']} completed | {d['upcoming']} upcoming | {d['teams']} teams | xG: {'yes' if d['xg_available'] else 'no'} | SOT: {'yes' if d['shots_available'] else 'no'}")

    def fixture_selected(self, _event=None):
        chosen = self.fixture_var.get()
        if chosen in self.fixture_map:
            h, a, _ = self.fixture_map[chosen]
            self.home_var.set(h)
            self.away_var.set(a)

    def predict(self):
        if self.df is None:
            messagebox.showwarning('No data', 'Upload one or more CSV files first.')
            return
        home, away = self.home_var.get(), self.away_var.get()
        ts = None
        chosen = self.fixture_var.get()
        if chosen in self.fixture_map:
            fh, fa, fts = self.fixture_map[chosen]
            if fh == home and fa == away:
                ts = fts
        try:
            self.result, self.training = predict_fixture(self.df, home, away, ts)
        except Exception as e:
            messagebox.showerror('Prediction error', str(e))
            return
        self._render_all()
        self.tabs.select(0)

    def _render_all(self):
        r, training = self.result, self.training
        pick = max([(r.home_team, r.p_home), ('Draw', r.p_draw), (r.away_team, r.p_away)], key=lambda x: x[1])
        confidence = max(r.p_home, r.p_draw, r.p_away)
        quality = 'High' if r.data_quality >= .75 else 'Moderate' if r.data_quality >= .50 else 'Limited'
        pred = [
            f'{r.home_team} vs {r.away_team}',
            '=' * 84,
            f'Data cutoff / kickoff: {r.kickoff}',
            f'Training matches: {r.training_matches} | Connected group: {r.group_size} teams | Data quality: {quality} ({r.data_quality*100:.0f}%)',
            '',
            'FINAL 1X2 PROBABILITIES',
            '-' * 84,
            f'1  {r.home_team:<30} {r.p_home*100:6.2f}%',
            f'X  Draw{"":<27} {r.p_draw*100:6.2f}%',
            f'2  {r.away_team:<30} {r.p_away*100:6.2f}%',
            '',
            f'PREDICTION: {pick[0]} ({pick[1]*100:.1f}%)',
            f'Expected goals: {r.home_team} {r.lambda_home:.2f} - {r.lambda_away:.2f} {r.away_team}',
            f'Highest single-outcome probability: {confidence*100:.1f}%',
            '',
            'MOST LIKELY SCORELINES',
        ]
        pred += [f'  {s:<6} {p*100:5.2f}%' for s,p in r.top_scores]
        pred += ['', 'Notes:'] + [f'• {n}' for n in r.notes]
        self._set(self.pred_text, '\n'.join(pred))

        def fmt(s):
            return f"M {s['Matches']:>2} | W-D-L {s['W']}-{s['D']}-{s['L']} | GF {s['GF']:.2f} | GA {s['GA']:.2f} | PPG {s['PPG']:.2f} | GD/m {s['GD/Match']:.2f} | xGF {s['xGF']} | xGA {s['xGA']}"
        h_over = team_summary(training, r.home_team)
        h5 = team_summary(training, r.home_team, last_n=5)
        hh5 = team_summary(training, r.home_team, venue='H', last_n=5)
        a_over = team_summary(training, r.away_team)
        a5 = team_summary(training, r.away_team, last_n=5)
        aa5 = team_summary(training, r.away_team, venue='A', last_n=5)
        strength = [
            'LEAGUE-RELATIVE ATTACK / DEFENCE PROFILE', '='*125,
            '100 = comparison-group average for the indices. Strength/weakness values are weighted component-deviation points versus average and can both be non-zero.',
            '',
            f'{"Team":<24} {"Atk Idx":>8} {"Def Idx":>8} {"Atk Str":>11} {"Atk Weak":>11} {"Def Str":>11} {"Def Weak":>11} {"DefWeak Idx":>12}',
            '-'*125,
            f'{r.home_team:<24} {r.home_attack:8.1f} {r.home_defense:8.1f} {r.home_attack_strength:11.1f} {r.home_attack_weakness:11.1f} {r.home_defense_strength:11.1f} {r.home_defense_weakness:11.1f} {r.home_def_weakness:12.1f}',
            f'{r.away_team:<24} {r.away_attack:8.1f} {r.away_defense:8.1f} {r.away_attack_strength:11.1f} {r.away_attack_weakness:11.1f} {r.away_defense_strength:11.1f} {r.away_defense_weakness:11.1f} {r.away_def_weakness:12.1f}',
            '',
            'ATTACK COMPONENT INDICES (100 = average; higher = stronger)',
            f'{r.home_team:<24} ' + ' | '.join(f'{k}: {v*100:.1f}' if pd.notna(v) else f'{k}: N/A' for k,v in r.home_attack_components.items()),
            f'{r.away_team:<24} ' + ' | '.join(f'{k}: {v*100:.1f}' if pd.notna(v) else f'{k}: N/A' for k,v in r.away_attack_components.items()),
            '',
            'DEFENSIVE VULNERABILITY COMPONENT INDICES (100 = average; higher = worse)',
            f'{r.home_team:<24} ' + ' | '.join(f'{k}: {v*100:.1f}' if pd.notna(v) else f'{k}: N/A' for k,v in r.home_defense_components.items()),
            f'{r.away_team:<24} ' + ' | '.join(f'{k}: {v*100:.1f}' if pd.notna(v) else f'{k}: N/A' for k,v in r.away_defense_components.items()),
            '', f'{r.home_team} — overall:       {fmt(h_over)}',
            f'{r.home_team} — last 5 overall:{fmt(h5)}',
            f'{r.home_team} — last 5 HOME:   {fmt(hh5)}',
            '', f'{r.away_team} — overall:       {fmt(a_over)}',
            f'{r.away_team} — last 5 overall:{fmt(a5)}',
            f'{r.away_team} — last 5 AWAY:   {fmt(aa5)}',
            '', 'Interpretation:',
            '• Attack Index >100 = stronger scoring/chance-creation profile than group average.',
            '• Defence Index >100 = stronger resistance to conceding than group average.',
            '• Attack/Defence Strength and Weakness separately sum favourable and unfavourable component deviations versus average, so both can be non-zero.',
            '• Defensive Weakness Index >100 = more vulnerable than group average; lower is better.'
        ]
        self._set(self.str_text, '\n'.join(strength))

        explanation = ['MATCH-SPECIFIC EXPLANATION', '='*100, '']
        for line in build_match_explanation(r, training):
            explanation.append('• ' + line)
            explanation.append('')
        self._set(self.explain_text, '\n'.join(explanation))

        hform = recent_matches_table(training, r.home_team, n=5)
        hhform = recent_matches_table(training, r.home_team, venue='H', n=5)
        aform = recent_matches_table(training, r.away_team, n=5)
        aaform = recent_matches_table(training, r.away_team, venue='A', n=5)
        form = [f'{r.home_team} — LAST 5 OVERALL', hform.to_string(index=False), '',
                f'{r.home_team} — LAST 5 HOME', hhform.to_string(index=False), '',
                f'{r.away_team} — LAST 5 OVERALL', aform.to_string(index=False), '',
                f'{r.away_team} — LAST 5 AWAY', aaform.to_string(index=False)]
        self._set(self.form_text, '\n'.join(form))

        rows = [
            ('Poisson / Dixon-Coles', r.poisson_probs),
            ('Elo', r.elo_probs),
            ('Recent Form', r.form_probs),
            ('Attack-Defence', r.strength_probs),
            ('FINAL ENSEMBLE', (r.p_home, r.p_draw, r.p_away)),
        ]
        model = ['MODEL                    HOME       DRAW       AWAY', '-'*62]
        for name, probs in rows:
            model.append(f'{name:<24} {probs[0]*100:7.2f}%   {probs[1]*100:7.2f}%   {probs[2]*100:7.2f}%')
        model += ['', 'ENSEMBLE WEIGHTS'] + [f'• {k}: {v*100:.0f}%' for k,v in r.model_weights.items()]
        self._set(self.model_text, '\n'.join(model))

    def _set(self, widget, text):
        widget.delete('1.0','end')
        widget.insert('1.0', text)

    def _method_text(self):
        return """SOCCER PREDICTOR — METHOD

1) CSV parsing
The app accepts one or more CSV files and maps common soccer column names such as HomeTeam/AwayTeam/FTHG/FTAG to the internal format. If xG and shots-on-target are present they are used; otherwise the model falls back toward league-average values for those missing components.

2) Team attack and defence
Attack = 55% goals scored + 30% xG + 15% shots on target.
Defensive vulnerability = 55% goals conceded + 30% xGA + 15% opponent shots on target.
If an optional xG/SOT metric is absent, its weight is redistributed across the metrics that are present.

Each component blends:
• 30% season overall
• 25% season home/away
• 25% last 5 overall
• 20% last 5 in the relevant venue split
Small samples are Bayesian-shrunk toward the competition average.

3) Four prediction models
• Poisson / Dixon-Coles score model
• Elo team-rating model
• Recent-form model (PPG, goal difference, xG difference)
• Attack-v-defence logistic strength model

4) Final 1X2 ensemble
• 60% Poisson / Dixon-Coles
• 20% Elo
• 15% recent form
• 5% attack-defence model

5) Strength / weakness interpretation
• Attack Index: 100 = average; above 100 is better.
• Defence Index: 100 = average; above 100 is better.
• Attacking Strength / Weakness: weighted favourable/unfavourable attack-component deviations versus average.
• Defensive Strength / Weakness: weighted favourable/unfavourable defensive-component deviations versus average.

6) Validation status
The 60/20/15/5 ensemble is a preliminary default informed by a walk-forward check on the supplied Finland history. Feature and recency weights remain fixed rules, and the probabilities are not yet universally calibrated. A stronger production model should optimise and calibrate them per competition using log loss, Brier score and 1X2 calibration while preserving strict pre-kickoff data cutoffs.

7) Leakage protection
For an upcoming fixture already present in the CSV, only matches completed before that kickoff are used. For a manually created pairing, the latest completed match in the uploaded data is used as the cutoff.

No bookmaker odds are required or used.
"""

if __name__ == '__main__':
    SoccerPredictorApp().mainloop()
