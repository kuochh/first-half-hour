# first-half-hour

**Can the last half-hour effect be made stronger by picking the days?**

Two *Journal of Financial Economics* papers (Gao, Han, Li and Zhou 2018; Baltussen, Da, Lammers and Martens 2021)
found that the last half-hour of the trading day tends to follow the direction of the day so far. Both report an
average over all days. This project:

1. rebuilds both papers' results on one-minute SPY data;
2. trains a model that scores each day, before the bet is placed, on whether following the day's direction will pay
   after costs, and compares "follow only the top-scored days" with "follow every day";
3. repeats this on other ETFs (international equity, bonds, gold).

Status: work in progress. Results, the write-up and an interactive app will appear here.

## Data

Minute bars come from a licensed vendor and are **not** included. `data/README.md` explains how to obtain the same
data with your own key and how to check that it matches. Only derived results are published.

## Licence

MIT. Developed with AI assistance (Claude); every analysis choice and result is reviewed by the author.
