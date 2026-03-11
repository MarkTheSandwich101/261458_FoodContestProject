# ต้องมีโฟลเดอร์ Intragram Photos ในนั้นจะมีโฟลเดอร์ย่อยที่แบ่งภาพออกเป็น 5 เมนู ได้แก่ Burger , Dessert , Pizza , Ramen และ Sushi
# ต้องมีโฟลเดอร์ Questionaire Images ในนั้นจะมีโฟลเดอร์ย่อยที่แบ่งภาพออกเป็น 5 เมนู ได้แก่ Burger , Dessert , Pizza , Ramen และ Sushi
# ต้องมีโฟลเดอร์ Test Images ในนั้นจะมีภาพเมนูอาหารเพื่อใช้ในการทำการแข่งขันหาอาหารน่ากิน

# Libraries ที่ใช้
import os # จัดการเกี่ยวกับไฟล์ / โฟลเดอร์ในคอมพิวเตอร์
import numpy as np # คำนวณตัวเลข / จัดการข้อมูลแบบ Array (เมทริกซ์)
import pandas as pd # จัดการข้อมูลที่เป็นตาราง (เหมือนใช้ Excel แต่เขียนด้วยโค้ด)
import tensorflow as tf # ตัวสร้าง AI (Deep Learning)
from tensorflow.keras.preprocessing.image import load_img, img_to_array # ตัวโหลด / แปลงรูปเป็นตัวเลข
from tensorflow.keras.models import Model # ตัวสร้างโครงสร้างสมอง AI
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Subtract # ชิ้นส่วนต่างๆ ของสมอง AI (เหมือนเลโก้)
from tensorflow.keras.applications import MobileNetV2 # โมเดล AI สำเร็จรูปที่เก่งเรื่องการดูรูป

print("\n")
print("Starting TensorFlow version:", tf.__version__)

# กำหนดที่อยู่โฟลเดอร์หลักที่เราเก็บงานไว้ (แก้ให้ตรงกับเครื่องของตัวเอง)
WORKSPACE_PATH = r"C:\Users\USER\Desktop\FoodContestProject"

# ==========================================
# STEP1: โหลดข้อมูล CSV
# ==========================================
print("Loading Files CSV... \n")
# อ่านไฟล์ข้อมูลการโหวตจากแบบสอบถาม data_from_questionaire.csv (ว่ารูปไหนชนะ)
train_df = pd.read_csv(os.path.join(WORKSPACE_PATH, 'data_from_questionaire.csv'))
# แปลงผลลัพธ์ให้คอมพิวเตอร์เข้าใจ: ถ้าคนโหวตเลข 1 (รูปซ้ายชนะ) ให้เก็บค่าเป็น 0, ถ้าคนโหวตเลข 2 (รูปขวาชนะ) ให้เก็บค่าเป็น 1
train_df['target'] = train_df['Winner'].apply(lambda x: 0 if x == 1 else 1)

# กำหนดขนาดรูปที่ AI ตัวนี้รับได้ (กว้าง 224 x สูง 224 x สี RGB 3 แชนแนล)
IMG_SHAPE = (224, 224, 3)
# กำหนดจำนวนรูปที่จะส่งให้ AI เรียนในแต่ละรอบ (ไม่ให้คอมพิวเตอร์ค้าง)
BATCH_SIZE = 16

# ==========================================
# STEP2: สร้างตัวป้อนรูปให้ AI (Data Generator)
# ==========================================
# สร้างคลาส (ตัวแทน) ที่มีหน้าที่ไปหยิบรูปมาป้อนให้ AI แบบอัตโนมัติ
class SiameseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, image_dir, batch_size, img_shape, is_training=True):
        super().__init__()
        self.dataframe = dataframe # ตารางข้อมูล CSV
        self.image_dir = image_dir # ชื่อโฟลเดอร์รูป
        self.batch_size = batch_size # จำนวนรูปต่อรอบการเรียนรู้
        self.img_shape = img_shape # ขนาดรูป
        self.is_training = is_training # เช็คว่าตอนนี้กำลัง 'Train' หรือแค่ 'Test'
        self.indexes = np.arange(len(self.dataframe)) # สร้างเลขลำดับเพื่อใช้สุ่มหยิบข้อมูล
        self.base_path = WORKSPACE_PATH
        
    def __len__(self):
        # คำนวณว่าต้องป้อนข้อมูลกี่รอบถึงจะครบ 1 จบ (Epoch)
        return int(np.ceil(len(self.dataframe) / self.batch_size))
    
    def on_epoch_end(self):
        # สลับลำดับข้อมูลทุกครั้งที่เรียนจบ 1 รอบ AI จะได้ไม่แอบท่องจำลำดับคำตอบ
        if self.is_training:
            np.random.shuffle(self.indexes)
            
    def load_and_preprocess(self, img_name, menu_folder=None):
        # ฟังก์ชันสำหรับโหลดรูปภาพ / แต่งสีก่อนส่งให้ AI
        # ถ้ามีระบุหมวดหมู่ (ตอน Train) ให้เข้าโฟลเดอร์เมนู (เช่น Pizza, Sushi) ก่อน
        if menu_folder:
            img_path = os.path.join(self.base_path, self.image_dir, menu_folder, img_name)
        # ถ้าไม่มีระบุหมวดหมู่ (ตอน Test) ให้ดึงรูปจากโฟลเดอร์หลักเลย
        else:
            img_path = os.path.join(self.base_path, self.image_dir, img_name)
            
        # โหลดรูป / ย่อขยายให้ได้ขนาด 224x224 ตามที่กำหนดไว้
        img = load_img(img_path, target_size=self.img_shape[:2])
        # แปลงรูปภาพให้เป็นชุดตัวเลข (Array)
        img_array = img_to_array(img)
        # ปรับสเกลสีของรูปภาพให้เข้ากับสิ่งที่โมเดล MobileNetV2 คุ้นเคย
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    def __getitem__(self, index):
        # ฟังก์ชันสำหรับจัดเตรียมรูปภาพเป็นกลุ่มๆ (Batch) เพื่อส่งให้ AI
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        # เตรียมกล่องเปล่าไว้ใส่ รูปที่ 1, รูปที่ 2 และ คำตอบ (ว่าใครชนะ)
        img1_batch, img2_batch, labels_batch = [], [], []
        
        # วนลูปตามตาราง CSV เพื่อหยิบรูปมาใส่กล่องทีละคู่
        for _, row in batch_df.iterrows():
            if self.is_training:
                # ถ้ากำลังเรียนรู้: ให้อ่านคอลัมน์ 'Menu' (เช่น Sushi) เพื่อเข้าโฟลเดอร์ให้ถูก
                menu = row['Menu']
                img1_batch.append(self.load_and_preprocess(row['Image 1'], menu))
                img2_batch.append(self.load_and_preprocess(row['Image 2'], menu))
                labels_batch.append(row['target']) # เก็บคำตอบเฉลยไว้ด้วย
            else:
                # ถ้ากำลัง Test: รูปจะถูกดึงตรงๆ จากโฟลเดอร์ Test Images โดยไม่มีหมวดหมู่
                img1_batch.append(self.load_and_preprocess(row['Image 1']))
                img2_batch.append(self.load_and_preprocess(row['Image 2']))
                
        # แพ็คกล่องเตรียมส่ง: ใส่ () เพื่อให้ Keras 3 มองว่าเป็นก้อนข้อมูลเดียวกัน (Tuple)
        if self.is_training:
            # ถ้าเรียนรู้: ส่งไปทั้งคำถาม (รูป1,รูป2) และ คำตอบเฉลย
            return (np.array(img1_batch), np.array(img2_batch)), np.array(labels_batch)
        # ถ้าทำ Test: ส่งไปแค่คำถาม (รูป1,รูป2) เพื่อให้ AI เดาคำตอบเอง 
        # (ต้องซ้อนวงเล็บ 2 ชั้น เพื่อบอกระบบว่านี่คือ 1 กลุ่มคำถาม)
        return ((np.array(img1_batch), np.array(img2_batch)), )

# สั่งให้ตัว Generator เตรียมโหลดรูปภาพจากโฟลเดอร์ที่กำหนด
train_gen = SiameseDataGenerator(train_df, image_dir='Questionaire Images', batch_size=BATCH_SIZE, img_shape=IMG_SHAPE)

# ==========================================
# STEP3: สร้างสมอง AI แบบฝาแฝด (Siamese Network Model)
# ==========================================
print("Building Siamese Network Model ... \n")
# เอาสมองสำเร็จรูป MobileNetV2 มาใช้เป็น 'ตา' สำหรับมองรูปอาหาร
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
# ล็อก 'ตา' นี้ไว้ไม่ให้เปลี่ยนความจำเดิม จะได้ประหยัดแรงคอมพิวเตอร์ / เรียนรู้เร็วขึ้น
# Freeze Weights โดยการล็อกน้ำหนักของ MobileNetV2 ไม่ให้เปลี่ยนค่าตอน Train กันอาการ Overfitting
base_model.trainable = False 

# สร้างระบบสกัดจุดเด่น (Feature Extractor)
inputs = Input(shape=IMG_SHAPE)
x = base_model(inputs)
x = GlobalAveragePooling2D()(x) # บีบอัดข้อมูลรูปภาพที่ซับซ้อนให้เป็นชุดตัวเลขสั้นๆ
feature_extractor = Model(inputs, x, name="feature_extractor")

# สร้าง 'ทางเข้า' 2 ทาง สำหรับรับรูปภาพ 2 รูปมาเทียบกัน
img1_in = Input(shape=IMG_SHAPE, name="Image_1")
img2_in = Input(shape=IMG_SHAPE, name="Image_2")

# ให้รูปภาพทั้ง 2 วิ่งผ่าน ระบบสกัดจุดเด่น อันเดียวกัน เพื่อการตัดสินแบบแฟร์ๆ
feat1 = feature_extractor(img1_in)
feat2 = feature_extractor(img2_in)

# นำจุดเด่นของรูปทั้ง 2 มา 'ลบกัน' เพื่อหาว่ารูปไหนเด่นกว่ากัน
diff = Subtract()([feat1, feat2])

# สร้างส่วน 'สมองตัดสินใจ' หลังจากรู้ความแตกต่างแล้ว
x = Dense(64, activation='relu')(diff)
# ตัดสินผลลัพธ์สุดท้าย: ค่าเข้าใกล้ 0 คือรูป 1 ชนะ, ค่าเข้าใกล้ 1 คือรูป 2 ชนะ
output = Dense(1, activation='sigmoid', name="output")(x)

# ประกอบร่างทุกส่วนเข้าด้วยกันเป็น AI 1 ตัว
model = Model(inputs=[img1_in, img2_in], outputs=output)
# ตั้งค่าวิธีการเรียนรู้ (ใช้ Adam Optimizer) และวิธีวัดความผิดพลาด (Binary_Crossentropy)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# STEP4: เริ่มสอน AI (Training)
# ==========================================
print("\n")
print("Training... \n")
# สั่งให้ AI เริ่มเรียนรู้จากข้อมูล 5 รอบ (Epochs)
history = model.fit(train_gen, epochs=5) 

# ==========================================
# สเตปที่ 5: ให้ AI ทำ test จริงด้วยไฟล์ test.csv 
# ==========================================
print("Test with test.csv... \n")
# โหลด test.csv มาเตรียมไว้
test_df = pd.read_csv(os.path.join(WORKSPACE_PATH, 'test.csv'))

# เตรียมตัวป้อนรูปภาพสำหรับสอบ (ปิดโหมดเรียนรู้ is_training=False)
test_gen = SiameseDataGenerator(test_df, image_dir='Test Images', batch_size=BATCH_SIZE, img_shape=IMG_SHAPE, is_training=False)

# สั่งให้ AI ทำนายผลว่ารูปไหนชนะ
predictions = model.predict(test_gen)
# แปลงตัวเลขทศนิยมที่ AI ตอบ กลับไปเป็นเลข 1 กับ 2
test_df['Winner'] = [1 if p < 0.5 else 2 for p in predictions]

# บันทึกคำตอบลงในไฟล์ CSV ตัวใหม่
output_path = os.path.join(WORKSPACE_PATH, 'test_result_ready_to_submit.csv')
test_df.to_csv(output_path, index=False)
print(f"Predicted! \n Saved In: {output_path} \n")

# บันทึกสมอง AI (โมเดล) ทั้งก้อนเอาไว้
model_path = os.path.join(WORKSPACE_PATH, 'my_siamese_model.keras')
model.save(model_path)
print(f"Saved Model at: {model_path} \n")
