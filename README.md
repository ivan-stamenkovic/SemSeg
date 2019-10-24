# SemSeg - Uputstvo za upotredu

## Direktorijumi

Projekat očekuje sledeću direktorijumsku strukturu:
* **dataset** - Folder za sve ulazne i izlazne slike u toku treninga  
  * **raw** - Ovde stoje neobrađene slike iz skupa podataka koje sam dobio  
    * **image** - Ulazne slike iz skupa podataka  
    * **gt_s** - Maske stakla iz skupa podataka  
  * **prepared** - Folder u koji skripta **image_preprocessor.py** snima obrađene ulazne slike. Ove slike se koriste u treningu  
    * **image** - Ulazne slike  
    * **gt_s** - Maske stakla  
  * **test** - Slike za test nakon treninga  
    * **image** - Ulazne slike
    * **output** - Folder u koji se smeštaju rezultati testa  
* **models** - Folder za istrenirane modele
  * **vgg** - Treba da sadrži pretrenirani vgg16 model, koji će se koristiti za kreiranje fcn8  
  * **fcn** - Skripta **model_trainer.py** će ovde smestiti istrenirani fcn8 model nakon što završi trening  
* **templates** - Folder za Flask templejte, sadrži samo **index.html**
* **tmp** - Folder u koji Flask server smešta ulazne i izlazne slike (čuvaju se bez posebnog razloga)  
  * **image** - Za čuvanje slika koje klijent pošalje
  * **output** - Za čuvanje slika nakon serverske obrade

## Skripte

Projekat sadrži sledeće skripte:
* **global_settings.py** - Čuva globalna podešavanja da ne bih morao da ih tražim u kodu  
* **image_preprocessor.py** - Vrši obradu ulaznih slika i maski tako da odgovaraju modelu (slike smanjujem u 512x384 jer se tensorflow buni da nema memoriju)  
* **model_runner.py** - Koristi mrežu da generiše krajnje slike  
* **model_trainer.py** - Kreiranje i obuka modela  
* **server_inference.py** - Wrapper za bilo šta što server treba da koristi, a ima veze sa mrežom  
* **server.py** - Flask server  

## Korišćenje

Projekat se koristi na sledeći način:
1. Instaliraju se paketi iz fajla **requirements.txt**  
2. Kreiraju se svi folderi opisani ranije i folderi **raw/image**, **raw/gt_s** i **test/image** se popune odgovarajućim slikama  
3. Prethodno istrenirani vgg16 model se smesti u **models/vgg**  
4. Pokrene se skripta **image_preprocessor.py**, koja generiše pripremljene slike  
5. Ako je sve prošlo uspešno, pokrene se skripta **model_trainer.py**, koja kreira i trenira model. Ako sve prođe uspešno, snimljeni model se nalazi u **models/fcn**.  
6. Pokrene se skripta **server.py**, koja pokreće server  
7. Serveru se pristupa na localhost:5000 (note: prvi zahtev za obradu je sporiji nego ostali, dok se tensorflow inicijalizuje)