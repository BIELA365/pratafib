from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
from pathlib import Path

OUTPUT = Path(r"c:\Kuliah\semester 7\Prata\02_Referensi\Revier paper abiel_proposal.xlsx")

columns = [
    "No", "Kategori", "Bab", "Status", "Identitas", "Tujuan", "Metode",
    "Temuan utama", "Keterbatasan / saran", "Relevansi untuk proposal",
    "DOI / URL", "Catatan akses"
]

papers = [
    [1, "Epidemiologi AFib", "Bab I", "Dipakai", "Chugh et al. (2014). Worldwide Epidemiology of Atrial Fibrillation: A Global Burden of Disease 2010 Study. Circulation, 129, 837-847.", "Mengestimasi prevalensi, insidensi, morbiditas, dan mortalitas AFib secara global.", "Systematic review populasi dan pemodelan DisMod-MR pada 21 wilayah Global Burden of Disease.", "Beban AFib, prevalensi, insidensi, dan mortalitas meningkat pada periode 1990-2010.", "Data global memiliki heterogenitas antarwilayah dan tidak menggambarkan kondisi Indonesia secara langsung.", "Dasar paragraf pembuka tentang urgensi AFib global.", "https://doi.org/10.1161/CIRCULATIONAHA.113.005119", "Sumber peer-reviewed; gunakan angka dan tahun sesuai artikel."],
    [2, "Dampak klinis", "Bab I", "Dipakai", "Benjamin et al. (1998). Impact of Atrial Fibrillation on the Risk of Death: The Framingham Heart Study. Circulation, 98, 946-952.", "Menganalisis hubungan AFib dengan mortalitas setelah penyesuaian faktor risiko.", "Kohort Framingham dan analisis survival/logistic regression.", "AFib berhubungan dengan peningkatan mortalitas sekitar 1,5-1,9 kali.", "Kohort lama dan berasal dari populasi tertentu.", "Mendukung dampak serius AFib terhadap keselamatan pasien.", "https://doi.org/10.1161/01.CIR.98.10.946", "PDF tersedia di folder bab 1."],
    [3, "Prevalensi dan stroke", "Bab I", "Dipakai", "Go et al. (2001). Prevalence of Diagnosed Atrial Fibrillation in Adults: The ATRIA Study. JAMA, 285, 2370-2375.", "Mengestimasi prevalensi AFib dan implikasinya terhadap manajemen irama serta pencegahan stroke.", "Analisis data populasi dewasa dan diagnosis klinis AFib.", "Prevalensi meningkat pada usia lanjut dan AFib berkaitan dengan risiko stroke.", "Data berbasis diagnosis dan konteks Amerika Serikat.", "Mendukung urgensi deteksi dan screening AFib.", "https://doi.org/10.1001/jama.285.18.2370", "PDF tersedia di folder bab 1."],
    [4, "Tren epidemiologi", "Bab I", "Dipakai", "Schnabel et al. (2015). 50-Year Trends in Atrial Fibrillation Prevalence, Incidence, Risk Factors, and Mortality in the Framingham Heart Study. The Lancet, 386, 154-162.", "Menganalisis perubahan AFib selama lima dekade.", "Analisis kohort Framingham lintas periode waktu.", "Prevalensi dan insidensi AFib meningkat, sedangkan mortalitas setelah AFib menunjukkan perubahan antarperiode.", "Kohort berasal dari satu studi populasi dan tidak mewakili Indonesia.", "Mendukung argumen bahwa AFib merupakan masalah yang terus berkembang.", "https://doi.org/10.1016/S0140-6736(14)61774-8", "PDF tersedia di folder bab 1."],
    [5, "Data Indonesia", "Bab I", "Dipakai", "Karunawan, Pinzon, dan Siswanto (2021). The comparison of clinical features and clinical outcomes in stroke patients with atrial fibrillation and non-atrial fibrillation: a hospital-based stroke registry in Indonesia. European Heart Journal, 42, Supplement 1.", "Membandingkan karakteristik dan hasil klinis pasien stroke dengan dan tanpa AFib di Indonesia.", "Registry stroke rumah sakit Indonesia periode 2011-2018; analisis statistik klinis.", "Dari 8248 pasien stroke, 226 atau 2,7% mengalami AFib. Mortalitas AFib 20,8% dibanding 10,6% pada non-AFib.", "Bukan prevalensi nasional; hanya pasien stroke pada satu registry rumah sakit.", "Memberikan bukti lokal Indonesia tanpa mengklaim prevalensi populasi nasional.", "https://doi.org/10.1093/eurheartj/ehab724.2060", "Sumber Indonesia; gunakan istilah hospital-based registry."],
    [6, "Konteks Asia", "Bab I", "Dipakai", "Krittayaphong et al. (2025). The burden of atrial fibrillation and unmet needs of its management in Southeast Asia and South Asia. The Lancet Regional Health - Southeast Asia, 34, 100549.", "Membahas beban AFib dan kebutuhan pengelolaan di Asia Selatan dan Asia Tenggara.", "Review dan pembahasan regional berbasis studi klinis serta registry.", "Menunjukkan kebutuhan screening, tata kelola, dan akses layanan AFib di kawasan Asia.", "Tidak seluruh angka dapat dipakai sebagai angka spesifik Indonesia.", "Memperkuat konteks regional setelah data global dan Indonesia.", "https://doi.org/10.1016/j.lansea.2025.100549", "Paper regional terbaru."],
    [7, "Pedoman klinis", "Bab I", "Dipakai", "Hindricks et al. (2021). 2020 ESC Guidelines for the Diagnosis and Management of Atrial Fibrillation. European Heart Journal, 42, 373-498.", "Merangkum definisi, diagnosis, screening, dan manajemen AFib.", "Pedoman klinis ESC/EACTS berbasis tinjauan bukti.", "Menempatkan ECG dan evaluasi klinis sebagai dasar diagnosis AFib.", "Pedoman bukan studi pengembangan model PPG.", "Menjelaskan bahwa sistem PPG Anda adalah screening, bukan diagnosis klinis.", "https://doi.org/10.1093/eurheartj/ehaa612", "Referensi pedoman; cek pembaruan 2024 jika diperlukan."],
    [8, "Screening AFib", "Bab I", "Dipakai", "Freedman et al. (2017). Screening for Atrial Fibrillation: A Report of the AF-SCREEN International Collaboration. Circulation, 135, 1851-1867.", "Membahas dasar dan tantangan screening AFib.", "Consensus/review kolaborasi internasional.", "Screening perlu mempertimbangkan populasi, metode, validasi, dan tindak lanjut klinis.", "Tidak berfokus khusus pada sensor MAX30105.", "Mendukung alasan perlunya sistem screening wearable.", "https://doi.org/10.1161/CIRCULATIONAHA.116.026693", "Gunakan untuk argumen screening, bukan kinerja model."],
    [9, "Wearable AFib", "Bab I", "Dipakai", "Tison et al. (2018). Passive Detection of Atrial Fibrillation Using a Commercially Available Smartwatch. JAMA Cardiology, 3, 409.", "Mengembangkan dan memvalidasi deep neural network untuk deteksi AFib dari smartwatch.", "PPG smartwatch dan deep neural network; validasi pada populasi ambulatory.", "Menunjukkan kelayakan deteksi pasif AFib menggunakan wearable PPG.", "Perangkat, populasi, dan protokol berbeda dari ear-worn MAX30105.", "Jembatan dari kebutuhan wearable menuju PPG-based AFib detection.", "https://doi.org/10.1001/jamacardio.2018.0136", "PDF tersedia di folder bab 1."],
    [10, "Wearable PPG", "Bab I", "Dipakai", "Guo et al. (2019). Mobile Photoplethysmographic Technology to Detect Atrial Fibrillation. Journal of the American College of Cardiology, 74, 2365-2375.", "Mengevaluasi teknologi mobile PPG untuk screening AFib.", "Mobile photoplethysmography dengan pembanding ECG klinis.", "Mendukung penggunaan PPG mobile untuk screening AFib.", "Teknologi mobile tidak identik dengan ear-worn MAX30105.", "Mendukung dasar pemilihan PPG sebagai sinyal input.", "https://doi.org/10.1016/j.jacc.2019.08.019", "Detail halaman perlu dicek pada versi penerbit."],
    [11, "Wearable PPG", "Bab I", "Dipakai", "Yan et al. (2018). Contact-Free Screening of Atrial Fibrillation by a Smartphone Using Facial Pulsatile Photoplethysmographic Signals. Journal of the American Heart Association, 7.", "Menguji screening AFib menggunakan sinyal facial PPG dari kamera smartphone.", "Facial PPG dibandingkan dengan ECG dan fingertip PPG.", "Facial PPG dapat digunakan untuk mendeteksi irregularitas denyut yang berkaitan dengan AFib.", "Bukan sensor kontak dan bukan MAX30105.", "Bukti tambahan bahwa PPG dapat membawa informasi screening AFib.", "https://doi.org/10.1161/JAHA.118.008585", "PDF tersedia di folder bab 1."],
    [12, "Dasar PPG", "Bab II", "Dipakai", "Allen (2007). Photoplethysmography and its Application in Clinical Physiological Measurement. Physiological Measurement, 28, R1-R39.", "Menjelaskan prinsip kerja dan aplikasi klinis PPG.", "Review teori PPG, komponen DC/AC, sensor optik, dan aplikasi fisiologis.", "PPG dapat digunakan untuk memperoleh informasi denyut dan parameter fisiologis.", "Paper dasar dan tidak berfokus pada deep learning AFib.", "Landasan teori PPG.", "https://doi.org/10.1088/0967-3334/28/3/R01", "Referensi teori utama."],
    [13, "Wearable PPG", "Bab II", "Dipakai", "Charlton et al. (2022). Wearable Photoplethysmography for Cardiovascular Monitoring. Proceedings of the IEEE, 110, 355-381.", "Meninjau penggunaan wearable PPG untuk monitoring kardiovaskular.", "Review wearable PPG, kualitas sinyal, artefak, dan aplikasi klinis.", "Wearable PPG menjanjikan tetapi dipengaruhi motion artifact dan perbedaan lokasi sensor.", "Review luas, bukan eksperimen model 1D-TA-DSC.", "Mendukung teori wearable PPG dan domain gap klinis-wearable.", "https://doi.org/10.1109/JPROC.2022.3149673", "Referensi teori dan gap."],
    [14, "Hearable PPG", "Bab II", "Dipakai", "Azudin et al. (2023). The Principles of Hearable Photoplethysmography Analysis and Applications in Physiological Monitoring. Sensors, 23, 6484.", "Meninjau prinsip analisis PPG pada perangkat hearable.", "Review akuisisi dan analisis hearable PPG.", "Telinga merupakan lokasi potensial untuk akuisisi PPG, tetapi desain dan artefak perlu diperhatikan.", "Tidak secara khusus mengembangkan classifier AFib Anda.", "Dasar pemilihan konsep hybrid ear-worn.", "https://doi.org/10.3390/s23146484", "Referensi hearable PPG."],
    [15, "In-ear PPG", "Bab II", "Dipakai", "Adams et al. (2022). Accurate Detection of Heart Rate Using In-Ear Photoplethysmography in a Clinical Setting. Frontiers in Digital Health, 4, 909519.", "Menguji akurasi heart rate dari in-ear PPG.", "Akuisisi in-ear PPG pada kondisi klinis dan pembandingan dengan referensi.", "In-ear PPG berpotensi menghasilkan sinyal denyut yang akurat.", "Fokus pada heart rate, bukan klasifikasi AFib.", "Mendukung kelayakan lokasi telinga.", "https://doi.org/10.3389/fdgth.2022.909519", "Referensi ear-worn."],
    [16, "Ear-worn", "Bab II", "Dipakai", "Xue et al. (2025). PPG Earring: Wireless Smart Earring for Heart Health Monitoring. CHI 2025.", "Mengembangkan perangkat earring untuk monitoring kesehatan jantung.", "Desain wearable, akuisisi PPG, komunikasi perangkat, dan evaluasi usability.", "Menunjukkan potensi bentuk ear-worn untuk monitoring jantung.", "Tidak berfokus khusus pada AFib dan model 1D-TA-DSC.", "Mendukung motivasi desain ear-worn.", "https://doi.org/10.1145/3706598.3713856", "Referensi desain dan wearable."],
    [17, "Kenyamanan wearable", "Bab II", "Dipakai", "Shimura et al. (2022). Engineering the Comfort-of-Wear for Next Generation Wearables. Advanced Electronic Materials, 2200512.", "Membahas faktor kenyamanan dan pemakaian wearable.", "Analisis desain wearable dan comfort-of-wear.", "Kenyamanan merupakan faktor penting untuk pemantauan jangka panjang.", "Tidak spesifik PPG atau AFib.", "Mendukung kebutuhan desain wearable yang tidak mengganggu pengguna.", "https://doi.org/10.1002/aelm.202200512", "Referensi pendukung ear-worn."],
    [18, "Motion artifact", "Bab II", "Dipakai", "Zheng et al. (2024). Tiny-PPG: A Lightweight Deep Neural Network for Real-Time Detection of Motion Artifacts in Photoplethysmogram Signals on Edge Devices. Internet of Things, 25, 101007.", "Mendeteksi motion artifact pada PPG menggunakan model lightweight di edge device.", "Deep neural network ringan untuk klasifikasi kualitas/motion artifact PPG.", "Model ringan dapat digunakan secara real-time pada perangkat edge.", "Fokusnya motion artifact, bukan AFib langsung.", "Mendukung motivasi noise robustness dan edge AI.", "https://doi.org/10.1016/j.iot.2023.101007", "PDF tersedia di folder Jurnal_Paper."],
    [19, "Noise modeling", "Bab II", "Dipakai", "Bachi et al. (2023). ECG Modeling for Simulation of Arrhythmias in Time-Varying Conditions. IEEE Transactions on Biomedical Engineering, 70, 3449-3460.", "Membangun simulator arrhythmia dan noise yang berubah terhadap waktu.", "Discrete-time Markov chain, pemodelan arrhythmia, muscle noise, motion artifact, dan respiration.", "Simulator meningkatkan kompleksitas sinyal dan dapat membantu augmentasi machine learning.", "Model utama ECG, sehingga prinsipnya perlu diadaptasi secara hati-hati untuk PPG.", "Dasar pemodelan augmentasi AR dan motion artifact pada penelitian Anda.", "https://doi.org/10.1109/TBME.2023.3263475", "Gunakan sebagai dasar noise, bukan paper PPG langsung."],
    [20, "Dataset PPG", "Bab II", "Dipakai", "Moulaeifard et al. (2026). MIMIC-III-Ext-PPG, a PPG-based Benchmark Dataset for Cardiovascular and Respiratory Signal Analysis. Scientific Data, 13, 668.", "Menyediakan benchmark PPG untuk analisis cardiovascular dan respiratory.", "Ekstraksi dan anotasi sinyal PPG dari MIMIC-III Waveform Database.", "Menyediakan sumber PPG berlabel yang dapat digunakan sebagai data pengembangan.", "Sinyal berasal dari clinical bedside monitor, sehingga ada domain gap dengan MAX30105 ear-worn.", "Dasar pemilihan dataset dan pembahasan keterbatasan data.", "https://doi.org/10.1038/s41597-026-07335-8", "Perlu akses dan verifikasi versi dataset."],
    [21, "Noise / in-ear", "Bab II", "Dipakai", "Kim et al. (2026). Miniatured In-Ear Sensor for Motion Artifact-Free Real-Time Vital Sign Monitoring. IEEE Sensors Journal, 26.", "Mengembangkan perangkat in-ear untuk monitoring PPG, SpO2, dan heart rate.", "Sensor PPG reflektif, desain FPCB, material packaging, BLE, dan uji motion artifact.", "Desain in-ear dapat meningkatkan kestabilan akuisisi pada kondisi bergerak.", "Tidak fokus pada AFib atau klasifikasi deep learning.", "Mendukung hardware ear-worn dan masalah motion artifact.", "IEEE Sensors Journal, 2026; DOI perlu diverifikasi dari laman IEEE.", "PDF tersedia di folder Jurnal_Paper."],
    [22, "PPG arrhythmia", "Bab II", "Pendukung", "Han et al. (2025). Multiclass Arrhythmia Classification Using Multimodal Smartwatch Photoplethysmography and Other Signals.", "Menguji klasifikasi arrhythmia berbasis data smartwatch.", "Smartwatch PPG/multimodal dan machine learning/deep learning.", "Mendukung kemungkinan klasifikasi arrhythmia dari wearable PPG.", "Multimodal dan multiclass; tidak sama dengan PPG-only AFib binary.", "Paper pembanding, bukan dasar utama.", "DOI/metadata perlu dilengkapi dari sumber asli.", "Masukkan setelah metadata diverifikasi."],
]

excluded = [
    [1, "Murmu et al. (2024)", "PPG-to-ECG reconstruction", "Dikeluarkan dari inti", "Penelitian Anda tidak merekonstruksi ECG; PPG langsung masuk classifier AFib."],
    [2, "Zhu et al. (2021)", "ECG waveform reconstruction from PPG", "Dikeluarkan dari inti", "Tidak sesuai arah final PPG-only classification."],
    [3, "Tang et al. (2022)", "PPG-to-ECG reconstruction", "Dikeluarkan dari inti", "Tidak digunakan untuk sistem final."],
    [4, "Hussain et al. (2025)", "IoT, federated learning, blockchain, Jetson", "Cadangan", "Boleh dipakai untuk konteks edge security, tetapi IoT bukan ruang lingkup final."],
    [5, "Jiang et al. (2023)", "Multimodal neurocritical-care IoT", "Cadangan", "Terlalu jauh dari AFib PPG-only dan ear-worn."],
    [6, "Berger et al. (2023)", "GAN ECG synthesis", "Dikeluarkan", "Tidak diperlukan untuk preprocessing/training PPG saat ini."],
    [7, "Kardia Band paper", "Single-lead ECG smartwatch", "Cadangan Bab I", "Boleh mendukung wearable AF screening secara umum, bukan bukti PPG."],
    [8, "Rani et al. (2025/2026)", "IoT cardiovascular monitoring", "Cadangan", "Gunakan hanya untuk edge AI umum; hapus dari inti jika IoT tidak digunakan."],
]

bab1 = [[p[0], p[4], p[1], p[2], p[7], p[9], p[10]] for p in papers if p[2] == "Bab I"]
bab2 = [[p[0], p[4], p[1], p[2], p[7], p[9], p[10]] for p in papers if p[2] == "Bab II"]

sheets = {
    "Paper Review": (columns, papers),
    "Pemetaan Bab I": (["No", "Identitas", "Kategori", "Status", "Temuan yang dapat dikutip", "Peran dalam paragraf", "DOI / URL"], bab1),
    "Pemetaan Bab II": (["No", "Identitas", "Kategori", "Status", "Temuan yang dapat dikutip", "Peran dalam teori/metode", "DOI / URL"], bab2),
    "Dikeluarkan": (["No", "Paper", "Topik lama", "Status", "Alasan"], excluded),
    "Petunjuk": (["Bagian", "Keterangan"], [
        ["Tujuan workbook", "Review paper yang diselaraskan dengan proposal PPG-only AFib, ear-worn, noise robustness, dan edge AI."],
        ["Status Dipakai", "Referensi yang mendukung langsung Bab I atau Bab II."],
        ["Status Pendukung", "Relevan tetapi tidak menjadi dasar utama."],
        ["Dikeluarkan", "Paper tetap dicatat agar alasan perubahan ruang lingkup terdokumentasi."],
        ["Catatan", "Verifikasi metadata final, DOI, volume, halaman, dan akses sebelum daftar pustaka proposal."],
    ])
}


def col_letter(number):
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell(ref, value, style=0):
    text = escape(str(value if value is not None else ""))
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet(headers, rows, title):
    all_rows = [headers] + rows
    xml = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">', '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>', '<cols>']
    for index in range(1, len(headers) + 1):
        width = 18 if index in (1, 2, 3, 4) else 34
        if index in (6, 7, 8, 9, 10, 12):
            width = 52
        xml.append(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
    xml.append('</cols><sheetData>')
    for row_number, row in enumerate(all_rows, 1):
        xml.append(f'<row r="{row_number}">')
        for index, value in enumerate(row, 1):
            style = 1 if row_number == 1 else 0
            xml.append(cell(f'{col_letter(index)}{row_number}', value, style))
        xml.append('</row>')
    xml.append(f'</sheetData><autoFilter ref="A1:{col_letter(len(headers))}{len(all_rows)}"/></worksheet>')
    return ''.join(xml)


content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">', '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>', '<Default Extension="xml" ContentType="application/xml"/>', '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>', '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
for index in range(1, len(sheets) + 1):
    content_types.append(f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
content_types.append('</Types>')

workbook = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>']
for index, name in enumerate(sheets, 1):
    workbook.append(f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>')
workbook.append('</sheets></workbook>')

relationships = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>']
workbook_relationships = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
for index in range(1, len(sheets) + 1):
    workbook_relationships.append(f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>')
workbook_relationships.append(f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')

styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="10"/><name val="Calibri"/></font><font><b/><sz val="10"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf></cellXfs></styleSheet>'

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
    archive.writestr("[Content_Types].xml", ''.join(content_types))
    archive.writestr("_rels/.rels", ''.join(relationships))
    archive.writestr("xl/workbook.xml", ''.join(workbook))
    archive.writestr("xl/_rels/workbook.xml.rels", ''.join(workbook_relationships))
    archive.writestr("xl/styles.xml", styles)
    for index, (name, (headers, rows)) in enumerate(sheets.items(), 1):
        archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet(headers, rows, name))

print(OUTPUT)
