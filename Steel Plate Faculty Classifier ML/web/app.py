# Library
import os
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)


# ============================================ load model

model_path1 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model1.pkl')
model1 = joblib.load(model_path1)

model_path2 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model2.pkl')
model2 = joblib.load(model_path2)

model_path3 = os.path.join(os.path.dirname(__file__), 'models/steelplate_model3.pkl')
model3 = joblib.load(model_path3)


# ============================================ function

def process_input_data_steel_ml(request):
    # Update: Change the key names to match the input field names
    X_Minimum = request.form['X_MinimumSlider']
    X_Maximum = request.form['X_MaximumSlider']
    Y_Minimum = request.form['Y_MinimumSlider']
    Y_Maximum = request.form['Y_MaximumSlider']
    Pixels_Areas = request.form['Pixels_AreasSlider']
    X_Perimeter = request.form['X_PerimeterSlider']
    Y_Perimeter = request.form['Y_PerimeterSlider']
    Sum_of_Luminosity = request.form['Sum_of_LuminositySlider']
    Minimum_of_Luminosity = request.form['Minimum_of_LuminositySlider']
    Maximum_of_Luminosity = request.form['Maximum_of_LuminositySlider']
    Length_of_Conveyer = request.form['Length_of_ConveyerSlider']
    Steel_Plate_Thickness = request.form['Steel_Plate_ThicknessSlider']
    Empty_Index = request.form['Empty_IndexSlider']
    Type_of_Steel = request.form['Type_of_Steel']

    # 입력값을 데이터에 맞게 변환
    if Type_of_Steel == 'A300':
        Type_of_Steel = 0
    else:
        Type_of_Steel = 1

    # 개별 데이터를 NumPy 배열로 표현
    individual_data = np.array([X_Minimum, X_Maximum, Y_Minimum, Y_Maximum, Pixels_Areas,
                                X_Perimeter, Y_Perimeter, Sum_of_Luminosity, Minimum_of_Luminosity,
                                Maximum_of_Luminosity, Length_of_Conveyer, Steel_Plate_Thickness,
                                Empty_Index, Type_of_Steel], dtype=object)

    data_array = individual_data.reshape(1, -1)  # 개별 데이터를 2차원 배열로 변환
    data_array = np.array(data_array).astype(float)  # str 형태의 데이터를 float으로 변환
    return data_array


# steel_ml 예측 후 반환
def process_steel_input_ml():
    data_array = process_input_data_steel_ml(request)

    # 예측 수행
    prediction1 = model1.predict(data_array)

    # 타겟 리스트
    target_list = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

    if prediction1.tolist()[0] == 0 or prediction1.tolist()[0] == 5 or prediction1.tolist()[0] == 6 : # 예측값이 0, 5, 6인 경우
    
        # 2차 예측 수행
        data_array2 = np.delete(data_array, -2, axis=1)  # 'Empty Index' 컬럼 데이터 삭제
        prediction2 = model2.predict(data_array2) 

        if prediction2.tolist()[0] == 1 : # 2차 예측시 1인 경우

            # 3차 예측 수행
            prediction3 = model3.predict(data_array2) # data_array2 데이터 그대로 사용
            num = prediction3.tolist()[0]
            return(target_list[num])
                
        else : # 2차 예측시 0인 경우 3차 예측 거치지 않음
            num = prediction2.tolist()[0]
            return(target_list[num])

    else : # 1차 예측시 1, 2, 3, 4인 경우 2,3차 예측 거치지 않음
        num = prediction1.tolist()[0]
        return(target_list[num])



# ============================================ 페이지 구현

# home
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# steel plate ML
@app.route('/steel_page_ml', methods=['GET', 'POST'])
def steel_page_ml():
    if request.method == 'POST':
        return process_steel_input_ml()
    return render_template('steel_ml.html')

@app.route('/steel_predict_ml', methods=['POST'])
def steel_predict_ml():
    return process_steel_input_ml()


if __name__ == '__main__':
    app.run(debug=True)