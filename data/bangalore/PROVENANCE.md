# Bangalore field-count source

Sunil Kumar V and J. Ranjitha (2013), *Improvement of Traffic Operations in
Congested Signalized Intersections – A Case Study in Bangalore City*, IJERT
2(7), DOI 10.17577/IJERTV2IS70850.

Source: https://www.ijert.org/improvement-of-traffic-operations-in-congested-signalized-intersections-a-case-study-in-bangalore-city

`bilekahalli_2013_counts.csv` transcribes six movement counts from Table 4.
The paper identifies a 24-hour workday survey on 6 March 2013 using traffic
police video footage. Source page states CC BY 4.0. Attribution is retained.

Only daily vehicle counts are used. Conflicting hourly figures and computed
delay estimates elsewhere in the paper are excluded. No raw trajectories or
measured per-vehicle waiting times accompany these counts.

The generated benchmark divides daily totals by 24 to obtain a daily-average
hourly rate, then projects the six movement proportions onto twelve synthetic
corridors (each source movement is split equally across two corridor flows).
This is explicitly a **Bangalore-count-driven synthetic benchmark**. It is not
a reconstruction of the actual three-arm junction or a validated Bengaluru
nine-junction network. Counts permit demand-replay checks; they cannot validate
the simulated waiting-time improvements against real-world outcomes.
