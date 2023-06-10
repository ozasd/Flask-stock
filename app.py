from flask import *
from flask_cors import CORS

#---------------------------------------------
import model.StockData 
import model.MA
import ai.trainingData 
import ai.main
app = Flask(__name__, static_folder="./front-end/build", static_url_path='/')
CORS(app)


@app.route("/")
def index():
    return app.send_static_file('index.html')



#------------------------------------------------------------------
@app.route("/StockData", methods=["POST"])
def StockData():
    stock = request.form.get('stock')
    mo = request.form.get('mo')+"mo"
   
    try:
       df = model.StockData.get_yf(stock,mo)
           #js = df.to_json(orient = 'columns')
       if len(df) > 0:
           data = {"狀態": "成功", "資料": df}
           return jsonify(data)
       else:
           
           data = {"狀態": "無", "資料": df}
           return jsonify(data)
    except:
        data = {"狀態": "失敗", "輸入資料": stock}
        return jsonify(data)
    
    
@app.route("/GetMA", methods=["POST"])
def GetMA():
    stock = request.form.get('stock')
    mo = request.form.get('mo')+"mo"
    ma = int(request.form.get('ma'))
    
    try:
       df = model.MA.getMA(stock,mo,ma)
       
          #js = df.to_json(orient = 'columns')
       data = {"狀態": "成功", "資料": df}
       return jsonify(data)
    except:
        data = {"狀態": "失敗", "輸入資料": stock}
        return jsonify(data)
@app.route("/training", methods=["POST"])
def training():
    stock = request.form.get('stock')
    mo = request.form.get('mo')+"mo"
    df = ai.trainingData.main(stock_id = stock,period= mo)
    data = {"狀態": "成功", "驗證集精準度": df}
    return jsonify(data)
    
    try:
       df = ai.trainingData.main(stock_id = stock,period= mo)
       data = {"狀態": "成功", "驗證集精準度": df}
       return jsonify(data)
    except:
        data = {"狀態": "失敗", "輸入資料": stock}
        return jsonify(data)    

@app.route("/predictData", methods=["POST"])
def predictData():
    print('predictData')

    stock = request.form.get('stock')
    cash = float(request.form.get('cash'))
    results = ai.main.predict(stock,cash)
    data = {"狀態": "成功", "輸入資料": results}
    return jsonify(data)    
    try:
       results = ai.main.predict(stock,cash)
       data = {"狀態": "成功", "輸入資料": results}
       return jsonify(data)     
    except:
        data = {"狀態": "失敗", "輸入資料": stock}
        return jsonify(data)     



if __name__ == "__main__":
    import webbrowser
    webbrowser.open('http://localhost:80')

    #app.debug =True
    app.run(port=80)
    #app.run()


