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
15	Logistic regression: all three	0.878	18	1	4	Rejected