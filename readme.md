# AuthenticRAG — การปรับปรุงประสิทธิภาพการค้นคืนข้อมูลด้วย Cross-Encoder Reranking
 
## 1. ภาพรวมโครงการ (Project Overview)
 
โครงการนี้เป็นการศึกษาการปรับปรุงประสิทธิภาพการค้นคืนข้อมูล (Retrieval) ในระบบ Retrieval-Augmented Generation (RAG) โดยนำเทคนิค **Cross-Encoder Reranking** มาใช้จัดลำดับเอกสารที่เกี่ยวข้องกับคำถามให้มีความเหมาะสมมากขึ้น
 
พัฒนาต่อยอดจากโครงการต้นฉบับของอาจารย์:
 
GitHub - aekanun2020/2025-authenticRAG
 
**ขอบเขตการทดลอง:** มุ่งเน้นการวัดคุณภาพของการค้นคืนและจัดลำดับเอกสาร ยังไม่ได้ประเมินคุณภาพคำตอบสุดท้ายจาก LLM
 
---
 
## 2. วัตถุประสงค์ (Objectives)
 
1. พัฒนาระบบค้นคืนข้อมูลแบบ Hybrid Search โดยใช้ BM25 และ BGE-M3
2. รวมผลการค้นหาด้วย Reciprocal Rank Fusion (RRF)
3. เพิ่ม Cross-Encoder Reranking ด้วยโมเดล `BAAI/bge-reranker-v2-m3`
4. เปรียบเทียบประสิทธิภาพระหว่าง Baseline และ Improved
5. ประเมินผลด้วย Recall@5, MRR, nDCG@5 และ Latency
6. สร้าง Dashboard ด้วย Streamlit เพื่อแสดงผลการทดลอง
 
---
 
## 3. สถาปัตยกรรมระบบ (System Architecture)
 
### 3.1 Baseline — Hybrid Retrieval
 
User Question → BM25 + BGE-M3 Dense Retrieval → RRF Fusion → Top-5 Documents
 
### 3.2 Improved — Hybrid Retrieval + Reranking
 
User Question → BM25 + BGE-M3 Dense Retrieval → RRF Fusion → Top-20 Candidates → BGE Reranker v2 M3 → Top-5 Documents
 
### ความแตกต่างสำคัญ
 
**Baseline:** ใช้การรวมอันดับจาก Keyword Search และ Semantic Search
 
**Improved:** เพิ่ม Cross-Encoder เพื่อประเมินความเกี่ยวข้องระหว่างคำถามกับเอกสารแต่ละรายการอีกครั้ง ก่อนเลือกเอกสาร 5 อันดับแรก
 
---
 
## 4. ข้อมูลที่ใช้ในการทดลอง (Dataset)
 
| รายการ | รายละเอียด |
|---|---|
| ประเภทข้อมูล | ข้อมูลทางการแพทย์ภาษาไทย |
| เอกสารต้นฉบับ | 4 ไฟล์ Markdown |
| จำนวน Chunks | 53 |
| จำนวนคำถามทดสอบ | 4 คำถาม |
| Search Engine | OpenSearch (Local) |
| Embedding Model | BGE-M3 |
| Reranking Model | BGE Reranker v2 M3 |
| Relevant Documents | กำหนดด้วยผู้ประเมิน |
 
---
 
## 5. ผลการทดลอง (Experimental Results)
 
| Metric | Baseline | Improved |
|---|---:|---:|
| Recall@5 | 1.0000 | 1.0000 |
| MRR | 0.8750 | 1.0000 |
| nDCG@5 | 0.9127 | 1.0000 |
| Average Latency | 0.3467 วินาที | 4.7319 วินาที |
 
### 5.1 Recall@5
 
ทั้งสองระบบสามารถค้นพบเอกสารที่กำหนดว่าเกี่ยวข้องได้ครบภายใน 5 อันดับแรก จึงมีค่า Recall@5 เท่ากับ 1.0000
 
### 5.2 MRR (Mean Reciprocal Rank)
 
เพิ่มจาก **0.8750 เป็น 1.0000**
 
แสดงว่าเอกสารที่เกี่ยวข้องรายการแรกได้รับการจัดอันดับดีขึ้นโดยเฉลี่ย
 
### 5.3 nDCG@5
 
เพิ่มจาก **0.9127 เป็น 1.0000**
 
แสดงว่าระบบ Improved สามารถจัดเอกสารที่เกี่ยวข้องให้อยู่ในอันดับต้น ๆ ได้เหมาะสมขึ้นตามเกณฑ์การประเมิน
 
### 5.4 Latency
 
เพิ่มจาก **0.3467 เป็น 4.7319 วินาที**
 
เนื่องจากระบบต้องใช้โมเดล Cross-Encoder ประเมินเอกสารเพิ่มเติม
 
---
 
## 6. ตัวอย่างผลการปรับปรุง (Question-Level Analysis)
 
ตัวอย่างคำถาม Q01 เกี่ยวข้องกับอาการและสาเหตุของโรคกรดไหลย้อน
 
### Baseline
 
| Rank | Document | Relevant |
|---|---|---|
| 1 | doc_42 | No |
| 2 | doc_45 | Yes |
| 3 | doc_46 | No |
| 4 | doc_43 | Yes |
| 5 | doc_48 | No |
 
### Improved
 
| Rank | Document | Relevant |
|---|---|---|
| 1 | doc_45 | Yes |
| 2 | doc_43 | Yes |
| 3 | doc_42 | No |
| 4 | doc_48 | No |
| 5 | doc_47 | No |
 
**ผลการเปรียบเทียบ:** Reranker สามารถย้ายเอกสารที่เกี่ยวข้องมาอยู่ในอันดับ 1 และ 2 จากเดิมที่อยู่ในอันดับ 2 และ 4
 
---
 
## 7. ไฟล์สำคัญในโครงการ (Project Files)
 
| ไฟล์ | หน้าที่ |
|---|---|
| `local_baseline.py` | ระบบ Hybrid Retrieval |
| `test_retrieval.py` | ทดสอบการค้นคืนข้อมูล Baseline |
| `test_reranking.py` | ทดสอบ Cross-Encoder Reranking |
| `auto_evaluation.py` | ประเมินผลการทดลองอัตโนมัติ |
| `evaluation_dataset.json` | ชุดคำถามและเอกสารที่เกี่ยวข้อง |
| `evaluation_results.json` | ผลการประเมิน |
| `dashboard.py` | Dashboard แสดงผลด้วย Streamlit |
 
---
 
## 8. วิธีรันโปรแกรม (How to Run)
 
### 8.1 เตรียมระบบ
 
ก่อนเริ่มใช้งานต้องเตรียมสิ่งต่อไปนี้
 
- Python Environment และ Dependencies ที่จำเป็น
- Docker และ OpenSearch ที่กำลังทำงาน
- เอกสารที่นำเข้าและสร้างดัชนีใน OpenSearch แล้ว
- โมเดล Embedding และ Reranking
 
### 8.2 รันการประเมินผล
 
ใช้คำสั่ง:
 
    python auto_evaluation.py
 
### 8.3 เปิด Streamlit Dashboard
 
ใช้คำสั่ง:
 
    python -m streamlit run dashboard.py
 
เปิด Dashboard ผ่าน:
 
http://localhost:8501
 
Dashboard สามารถแสดงผลการเปรียบเทียบ Baseline และ Improved รวมถึงดาวน์โหลด PDF Evaluation Report ได้
 
---
 
## 9. ข้อจำกัดของการทดลอง (Limitations)
 
1. ใช้คำถามทดสอบเพียง 4 คำถาม จึงเป็นการประเมินเบื้องต้น
2. Relevant Documents ถูกกำหนดด้วยผู้ประเมิน และยังไม่ได้ตรวจสอบความครบถ้วนของเอกสารที่เกี่ยวข้องทั้งหมด
3. Baseline เป็น Local Hybrid Retrieval ที่ไม่ได้ใช้ Contextual Retrieval เต็มรูปแบบเหมือนโครงการต้นฉบับ
4. ยังไม่ได้ประเมินความถูกต้องของคำตอบสุดท้ายจาก LLM
5. การเพิ่ม Reranker ทำให้ระยะเวลาประมวลผลสูงขึ้น
 
---
 
## 10. สรุปผลการทดลอง (Conclusion)
 
จากการทดลองเบื้องต้นพบว่า **Cross-Encoder Reranking สามารถปรับปรุงคุณภาพการจัดลำดับเอกสารที่เกี่ยวข้องได้**
 
ผลการประเมินพบว่า:
 
- **Recall@5:** คงเดิมที่ 1.0000
- **MRR:** เพิ่มจาก 0.8750 เป็น 1.0000
- **nDCG@5:** เพิ่มจาก 0.9127 เป็น 1.0000
- **Latency:** เพิ่มจาก 0.3467 เป็น 4.7319 วินาที
 
การเพิ่ม Reranker จึงมีประโยชน์ต่อการจัดลำดับเอกสาร แต่ต้องพิจารณาต้นทุนด้านเวลาในการประมวลผลร่วมด้วย
 
อย่างไรก็ตาม ยังไม่สามารถสรุปได้ว่าคุณภาพคำตอบสุดท้ายของ LLM ดีขึ้น เนื่องจากการทดลองนี้ประเมินเฉพาะขั้นตอน Retrieval
 
---
 
## 11. แหล่งอ้างอิง (References)
 
### Original AuthenticRAG Repository
 
GitHub - aekanun2020/2025-authenticRAG
 
### Assignment Repository
 
GitHub - Nattakun30/authenticrag-reranking-evaluation: Hybrid RAG retrieval evaluation with BGE-M3 a…
 
โครงการนี้เป็นการพัฒนาต่อยอดจากแนวคิดและโค้ดต้นฉบับ โดยเพิ่มกระบวนการ Reranking, Evaluation และ Dashboard เพื่อศึกษาและเปรียบเทียบประสิทธิภาพการค้นคืนข้อมูล
