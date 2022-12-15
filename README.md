# PdfToXml

## Leírás
A magyar nyelvű digitális örökség nagyon nagy hányada javítatlan OCR formájában érhető el kétrétegű PDF fájlokban. Ezek olyan PDF fájlok, amelyekben az oldalak képe mögött helyezkedik el az OCR-es szöveg, ami kimásolható és kereshető. Azonban ez a réteg tagolatlan és nyers. A dokumentum tagolását csak a képpel egyeztetve tudhatjuk meg. Egyes eszközök képesek ezt a feladatot részben megoldani, de csak úgy, hogy az eredeti OCR réteg elvész. 

A cél egy olyan eszköz készítése mely a szöveget és annak alapvető tagolását képes felismerni ennek megfelelően egy XML fájlban elhelyezni azt (a pdf eredeti OCR rétegének elvesztése nélkül), úgy, hogy az adott szövegrészek a megfelelő XML tagek közé kerüljenek. 

Ehhez a feladathoz részben már léteznek nyílt eszközök. Ezekkel már elérhető szöveg felismerés, illetve layout felismerés. Ezen megoldásokat úgy egyesítve és továbbfejlesztve, készülne el a program, mely képes a felismert, formázatlan szöveget a layout egyes elemeihez hozzárendelni aszerint, hogy az hol helyezkedik el az adott oldalon.

Az így kapott XML már könnyen kezelhető, szerkeszthető, tetszőleges formátumba konvertálható lesz.

## Felhasznált könyvtárak
- tesseract-ocr
    - https://github.com/tesseract-ocr/tesseract
    - újra ocrezéshez
- pytesseract 
    - https://pypi.org/project/pytesseract/
- Pillow
    - https://python-pillow.org/
    - https://pypi.org/project/Pillow/
    - képek kezeléséhez
- PyMuPdf
    - pdf-ek feldolgozásához, képek exportálása, meglévő szövegek exportálása
    - https://pymupdf.readthedocs.io/en/latest/
    - https://pypi.org/project/PyMuPDF/
- pdf2image
    - másik megoldás képek exportálásához
    - https://pypi.org/project/pdf2image/

## Telepítési útmutató
### Windows
1. Telepítsük a Pythont legalább 3.9-es verzióját
2. Telepítsük fel a Tesseract-OCR-t és hozzá magyar nyelvi támogatást
    1. Töltsük le a telepítőt: https://github.com/UB-Mannheim/tesseract/wiki
    2. Indítsuk el az exe fájlt és kövessük a telepítő utasításait.
    3. A Choose components pontban, az Additional Language data lenyíló menüben pipáljuk be a Hungarian-t a magyar nyelv telepítéséhez.
    4. Folytassuk a telepítést
    5. e.	Ha végeztünk akkor adjuk hozzá a PATH környezeti változóhoz a telepítés útvonalát. (Ez alapértelmezetten C:\Program Files\Tesseract-OCR)
3. Telepítsük a popplert
    1. Töltsük le erről a linkről a legújabb release-t: https://github.com/oschwartz10612/poppler-windows/releases/
    2. Csomagoljuk egy tetszőleges helyre, majd adjuk hozzá a poppler/Library/bin mappát a PATH környezeti változóhoz.
4. Telepítsük a python könyvtárakat a következő parancsokkal
    ```
    pip install Pillow
    pip install pytesseract
    pip install pymupdf
    pip install pdf2image
    ```

## Használat
A szoftver jelenleg egyablakos asztali alkalmazásként érhető el. A felhasználó számára pedig a következő funkciók állnak rendelkezésre:
- Fájl megnyitása
- Mentés helyének kiválasztása
- Előnézetek lapozása
- A feldolgozás paramétereinek beállítása
- A feldolgozás elindítása

Nyissuk meg a programot. Ha megjelent az ablak, akkor kattintsunk a „Fájl megnyitása” gombra. A megjelenő ablakban válasszuk ki a feldolgozandó PDF-et. Ha ezzel megvagyunk, akkor kattintsunk a „megnyitás” gombra. Ezután kattintsunk a „Mentés helye” gombra. Ezután a megjelenő ablakban válasszuk ki, hogy hová szeretnénk menteni és nevezzük el a fájlt. A „mentés” gombra kattintva léphetünk tovább. A beállítások menü az induláskor alaphelyzetben van. Ha szeretnénk módosítsuk a megfelelő választó gombokra kattintva. Ezután kattintsunk az indítás gombra. Ekkor megkezdődik a fájl feldolgozása, amit egy állapotjelző sáv mutat nekünk. Amikor a program elkészült megjelenik egy üzenet a „Kész” felirattal.

### Fájl megnyitása
Kattintsunk a „Fájl megnyitása” gombra. A megjelenő ablakban válasszuk ki a feldolgozandó PDF-et. Ha ezzel megvagyunk, akkor kattintsunk a „megnyitás” gombra

### Mentés helyének kiválasztása
Kattintsunk a „Mentés helye” gombra. Ezután a megjelenő ablakban válasszuk ki, hogy hová szeretnénk menteni és nevezzük el a fájlt. A „mentés” gombra kattintva léphetünk tovább.

### Előnézetek lapozása
A programban lehetőség van előnézetek megtekintésére. A PDF-et bal oldalon láthatjuk, tőle jobbra pedig a kimeneti előnézeteket. A PDF alatti nyíl gombok segítségével lapozhatunk a PDF oldalain. Ezzel együtt lapoznak a kimeneti előnézetek is amik tartalmaz mindig az aktuális PDF oldalnak megfelelően kerül kiszámításra.

### Paraméterek beállítása
Lehetőségünk van beállítani, hogy milyen módon történjen a feldolgozás.
Jelenleg a következők közül választhatunk
- betöltési módok:
    - képek exportálása (a PDF-ben található képeket másolja ki, gyorsabb, de csak akkor működik megfelelően, ha oldalanként egy kép található)
    - oldalak leképezése (a teljes oldalt minden tartalommal együtt leképez, lassabb, de megbízhatóbb. Akkor válasszuk ezt ha nem megfelelően jelenik meg a PDF az előnézeten)
- OCR módok
    - eredeti (a PDF-ben található eredeti OCR-t használjuk fel)
    - Tesseract-OCR (újra OCR-ezzük a dokumentumot)

### Feldolgozás elindítása
Elöször nyissunk meg egy PDF-et és állítsuk be a mentés útvonalát. Ha ezekkel megvagyunk. akkor az indítás gombra kattintva indíthatjuk el a folyamatot.




