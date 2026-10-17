# Registro del lavoro

Una riga per ogni idea scartata (cosa, perché, numeri) e per ogni uso importante di un agente.


FEATURES PER CAPIRE LA PRESENZA DI UN PV - FLAVIO
#	What we tried	F1	TP	FP	FN	Verdict
0	Predict TRUE for everyone	0.710	22	18	0	Baseline
1	drop, first version (last 2 years only, no windows)	0.682	15	7	7	Rejected: below the baseline
2	drop (winter minus summer), on the windows	0.737	14	2	8	Rejected
3	winter_level	0.593	16	16	6	Rejected
4	duck (evening ramp-up)	0.714	15	5	7	Rejected: about the same as the baseline
5	step (lasting drop)	0.778	14	0	8	Rejected
6	var_summer (day-to-day swings)	0.780	16	3	6	Rejected
7	summer_level	0.783	18	6	4	Rejected
8	latest_summer	0.857	18	2	4	Kept only as a candidate
9	weekend_summer	0.889	20	3	2	Kept only as a candidate
10	dip_summer (midday against morning and afternoon)	0.894	21	4	1	Best single feature
11	clamp_summer (one clamping day is enough)	0.927	19	0	3	Rejected: fragile, relies on near-zero values
12	dip_summer + clamp tiebreaker (clamp needs ≥5% of summer days)	0.872	17	0	5	Rejected: worse than dip_summer alone
13	Logistic regression: dip_summer + weekend_summer	0.905	19	1	3	Best overall
14	Logistic regression: dip_summer + latest_summer	0.850	17	1	5	Rejected
15	Logistic regression: all three	0.878	18	1	4	Rejected16	dip_spread (p90 - p10 of the daily summer dip: PV dip comes and goes with clouds)	0.927	19	0	3	Kept: new main feature, LOO threshold stable 0.836-0.856; 1-year windows 0.900 (0 FP). Thin margin: max no-PV 0.83 vs threshold 0.85
17	dip_season (dip winter minus summer), dip_ext (Apr-Sep), dip_q25, dip_min_month, dip_recent_change	<=0.894	-	-	-	Rejected: not better than dip_summer
18	Logistic regression: dip_spread + dip_summer / weekend_summer / dip_season / clamp (C=1 and 0.1)	0.842-0.872	-	0-1	5-6	Rejected: more FN than dip_spread alone
19	OR rule: dip_spread high OR dip_summer very low (both thresholds fitted in LOO)	0.864	19	3	3	Rejected: second threshold unstable across folds
20	Agent (Claude Code, 2026-10-17): proposed and tested features 16-19 on per-window aggregates only; remaining FN are large plants (166, 178, 959 kWp), one installed 75 days before data end
