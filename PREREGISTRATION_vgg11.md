# Vorhersage — Wiederholung auf einer zweiten Architektur (VGG-11)

Festgehalten vor dem Start der Rechenlaeufe.

Aufbau: VGG-11 ohne Normalisierungsschicht, drei Bedingungen (RGB, CIELAB,
Graustufen), drei Startwerte (0, 1, 42), im Uebrigen identische
Trainingskonfiguration wie in Tabelle 3.

## V1 — Der Nullbefund bleibt bestehen
CIELAB und RGB unterscheiden sich auf mCA_Helligkeit nicht bedeutsam.
Begruendung: Die Filteranalyse zeigt, dass das Netz die Luminanz-Chrominanz-
Zerlegung selbst konstruiert; das ist keine Eigenschaft von ResNet.

## V2 — Der Trade-off bleibt bestehen
CIELAB verliert auf Rauschstoerungen und gewinnt auf Unschaerfestoerungen,
in derselben Richtung und derselben Groessenordnung.
Begruendung: Der Trade-off entsteht in der Farbraumtransformation und in der
Eingabenormalisierung, also VOR dem Netz. Er darf nicht von der Architektur
abhaengen. Verschwindet er, ist die Erklaerung aus Abschnitt 7.1 widerlegt.

## V3 — Die Negativkontrolle greift weiterhin
Graustufen liegt deutlich (> 4 Prozentpunkte) hinter RGB auf mCA.
Ohne diese Bestaetigung ist V1 nicht interpretierbar.

## V4 — Vorhersage aus Abschnitt 8.2.1 (nur bei norm="none")
Der Abstand zwischen den Farbraeumen auf mCA_Helligkeit ist groesser als unter
BatchNorm (0,81 pp) und mindestens so gross wie unter GroupNorm (2,49 pp).
Begruendung: Ohne Normalisierungsschicht entfaellt die Angleichung der
Kanalwertebereiche vollstaendig. Bleibt der Abstand geschlossen, ist die
Angleichung als Erklaerung ausgeschlossen.
