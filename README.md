# Dip reitingas

Automatinis intraday kritimo pirkimo skeneris. GitHub Actions paleidžia
`dip_reitingas.py` kas ~5 min. biržos valandomis ir atnaujina `docs/index.html`,
kurį galima peržiūrėti bet kuriame įrenginyje, įskaitant iPhone Safari,
per GitHub Pages nuorodą.

## Vienkartinis nustatymas

1. Šį aplanką (kartu su `.github` poaplankiu) įkelk kaip naują GitHub repozitoriją.
2. Settings → Pages → Source: **Deploy from a branch**, branch **main**, folder **/docs**.
3. Actions skirtuke paleisk "Dip reitingo atnaujinimas" rankiniu būdu (Run workflow)
   pirmam kartui — sukurs `docs/index.html`.
4. Atsidariusi nuoroda (rodoma Settings → Pages) atsidaro bet kur, be diegimo.
   iPhone: Safari → Share → "Add to Home Screen" — atrodys kaip programėlė.

Tada viskas vyksta savaime — GitHub serveris renka duomenis net kai
tavo kompiuteris ir telefonas išjungti.
