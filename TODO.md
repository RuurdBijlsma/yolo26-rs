# Status:

* run_onnx_v2.rs produceert goeie resultaten, zelfde als python run_onnx.py.
* de resize algo is super traag
* export_onnx2.py aan de python kant is nodig om de onnx te maken + de vocab json

# Todo:

* ✅ zie of ik de performance van run_onnx_v2 kan fixen. eerst maar zien waar de tijd precies in zit en dan [OPTIMIZEN].
* ✅ propere crate maken, inclusief mask enzo, met mooie visualisatie examples, en from_hf ding en bon builder enzo
* ✅ serde feature voor crate
* ✅ thiserror error handling
* ✅ execution providers toevoegen
* kleinere models pullen en kijken of die ook werken
* kan ik t ook werkend krijgen met class list als model input? Dus je zegt egg als input en dan detect ie alleen eggs
* leg in readme uit hoe je een onnx export in python
* ik wil graag text-prompt supporten, het liefst in rust only, kan dit?
    * ✅ maak eerst in python only
    * ✅ dan export onnx ervoor maken
    * ✅ dan in python onnx uitvoeren met ultralytics pre&post processing zien of de results matchen <- we are here, de
      results matchen niet
    * ✅ dan in python onnx uitvoeren zien of de results matchen
    * dan porten naar rust
        * de promptable_basic.rs example werkt soortvan, niet exact zelfde results als python onnx impl
        * maak mapje in src die promptable heet ofzo, en dan daarin de code zetten om 'm te runnen
        * maak example die visualized met mask+bbox+tags+score
        * zie of ik 'm gelijk kan krijgen met python-onnx
* benchmark alle model sizes en laat speed zien in readme
    * 26n: 169ms