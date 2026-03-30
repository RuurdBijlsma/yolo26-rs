# Status:

* run_onnx_v2.rs produceert goeie resultaten, zelfde als python run_onnx.py.
* de resize algo is super traag
* export_onnx2.py aan de python kant is nodig om de onnx te maken + de vocab json

# Todo:

* zie of ik de performance van run_onnx_v2 kan fixen. eerst maar zien waar de tijd precies in zit en dan [OPTIMIZEN].
* propere crate maken, inclusief mask enzo, met mooie visualisatie examples, en from_hf ding en bon builder enzo
* kleinere models pullen en kijken of die ook werken
* kan ik t ook werkend krijgen met class list als model input? Dus je zegt egg als input en dan detect ie alleen eggs