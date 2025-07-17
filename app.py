from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
import pickle
import numpy as np
import re
import threading
import time
import requests
import base64

import firebase_admin
from firebase_admin import credentials, messaging, firestore
from dotenv import load_dotenv
load_dotenv()

TRACKER_CLIENT_ID = os.environ.get("TRACKER_CLIENT_ID")
TRACKER_CLIENT_SECRET = os.environ.get("TRACKER_CLIENT_SECRET")

import json

FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS")  # 원문 JSON
firebase_creds = json.loads(FIREBASE_CREDENTIALS)               # 원문 -> dict
cred = credentials.Certificate(firebase_creds)                  # dict로 생성
firebase_admin.initialize_app(cred)

db = firestore.client()

SUBSCRIPTIONS_FILE = os.path.join('subscriptdata', 'subscriptions.json')

def load_subscriptions_from_file():
    global alert_subscriptions
    alert_subscriptions = []
    try:
        subscriptions_ref = db.collection("subscriptions")
        for doc in subscriptions_ref.stream():
            data = doc.to_dict()
            alert_subscriptions.append(data)
        print(f"📂 Firestore 구독 정보 로드 완료: {len(alert_subscriptions)}개")
    except Exception as e:
        alert_subscriptions = []
        print(f"❗ Firestore 구독 로드 실패: {e}")

def save_subscriptions_to_file():
    try:
        # 전체 alert_subscriptions를 Firestore로 저장
        for sub in alert_subscriptions:
            doc_ref = db.collection("subscriptions").document(f"{sub['user_id']}_{sub['invoice']}")
            doc_ref.set(sub)  # Firestore 저장
        print(f"☁️ Firestore 구독 정보 저장 완료: {len(alert_subscriptions)}개")
    except Exception as e:
        print(f"❗ Firestore 저장 실패: {e}")

def load_subscriptions_from_firestore():
    global alert_subscriptions
    try:
        alert_subscriptions = []
        subscriptions_ref = db.collection("subscriptions").stream()
        for doc in subscriptions_ref:
            data = doc.to_dict()
            alert_subscriptions.append(data)

        print(f"☁️ Firestore로부터 구독 로드 완료: {len(alert_subscriptions)}개의 구독")
    except Exception as e:
        print(f"❗ Firestore 구독 로드 실패: {e}")

# 🔐 Step 1. access_token 발급 함수
def get_access_token(client_id, client_secret):
    url = "https://auth.tracker.delivery/oauth2/token"
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()  # base64 인코딩
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = "grant_type=client_credentials"
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def predict_arrival_internal(status, last_time_str, carrier_id=None):
    try:
        normalized_status = normalize_status(status)

        # 🚀 carrier_id 기반 모델/매핑 불러오기
        model, status_map = load_model_and_mapping(carrier_id)
        if not model or not status_map:
            return {"status": "error", "message": "모델 또는 매핑 로드 실패"}

        code = status_map.get(normalized_status, -1)
        predicted_minutes = model.predict(np.array([[code]]))[0]

        last_time = datetime.fromisoformat(last_time_str)
        arrival_time = last_time + timedelta(minutes=predicted_minutes)

        base_date = arrival_time.date()
        # 🔔 일요일이면 하루 추가
        if base_date.weekday() == 6:
            base_date += timedelta(days=1)

        return {
            "status": "success",
            "predicted_minutes": round(predicted_minutes, 1),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

app = Flask(__name__)
CORS(app, origins=["https://alimbox.com"])

alert_subscriptions = []

# 🔧 상태 정규화 함수
def normalize_status(status):
    status = status.lower().strip()
    mapping_keywords = {
        '배송완료': ['배송완료', '배달완료', 'delivered'],
        '배송출발': ['배송출발', '배달출발', 'out for delivery'],
        '간선상차': ['간선상차', '캠프상차', '터미널상차', '상차'],
        '간선하차': ['간선하차', '캠프도착', '터미널하차', '하차'],
        '집화처리': ['접수', '인수', '소터분류', '운송장출력', '수거', '집하', '수집'],
        'sm 입고': ['입고', '센터입고'],
    }
    for norm_status, keywords in mapping_keywords.items():
        if any(kw in status for kw in keywords):
            return norm_status
    return status

def load_model_and_mapping(carrier_id):
    try:
        if carrier_id == 'kr.coupangls':
            with open('arrival_predictor_coupangls.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('status_mapping_coupangls.pkl', 'rb') as f:
                status_map = pickle.load(f)
        elif carrier_id == 'kr.epost':
            with open('arrival_predictor_epost.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('status_mapping_epost.pkl', 'rb') as f:
                status_map = pickle.load(f)
        elif carrier_id == 'kr.hanjin':
            with open('arrival_predictor_hanjin.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('status_mapping_hanjin.pkl', 'rb') as f:
                status_map = pickle.load(f)
        else:
            with open('arrival_predictor.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('status_mapping.pkl', 'rb') as f:
                status_map = pickle.load(f)
        return model, status_map
    except Exception as e:
        print(f"❗ 모델/매핑 파일 로드 실패: {e}")
        return None, None


@app.route('/test', methods=['GET'])
def test_api():
    return jsonify({'message': 'API 동작 확인 완료!', 'status': 'success'})

@app.route('/save_delivery', methods=['POST'])
def save_delivery():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'fail', 'message': 'JSON 데이터가 없습니다.'}), 400

        last_event = data.get('lastEvent', {})
        status_name = last_event.get('status', {}).get('name', '')
        normalized_status = normalize_status(status_name)
        invoice = data.get('invoice', 'unknown')
        carrier_id = data.get('carrier_id', 'unknown')

        print(f"📦 원본 상태: {status_name}")
        print(f"🔧 정규화 상태: {normalized_status}")
        print(f"🔎 lastEvent 전체 내용:\n{json.dumps(last_event, ensure_ascii=False, indent=2)}")

        if normalized_status != '배송완료':
            return jsonify({'status': 'ignored', 'message': '배송완료된 건만 저장합니다.'}), 200

        folder_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                with open(os.path.join(folder_path, filename), encoding='utf-8') as f:
                    existing = json.load(f)
                    if existing.get('invoice') == invoice:
                        return jsonify({'status': 'duplicate', 'message': f'{invoice}는 이미 저장된 송장번호입니다.'}), 200

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(folder_path, f'delivery_{timestamp}.json')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({'status': 'success', 'message': '배송 데이터 저장 완료!', 'file': file_path})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict_arrival', methods=['POST'])
def predict_arrival():
    try:
        data = request.get_json()
        status = data.get('status')
        last_time_str = data.get('last_time')
        carrier_id = data.get('carrier_id')  # carrier_id 받기
        print(f"🔔 [predict_arrival] 요청 carrier_id: {carrier_id}")

        if not status or not last_time_str:
            return jsonify({'status': 'fail', 'message': 'status 또는 last_time이 없습니다.'}), 400

        normalized_status = status.strip()

        # 🚀 carrier_id 기준으로 pkl 불러오기
        model, status_map = load_model_and_mapping(carrier_id)

        if not model or not status_map:
            return jsonify({'status': 'fail', 'message': '모델 또는 매핑 로드 실패'}), 500

        if normalized_status not in status_map:
            print(f"⚠️ 알 수 없는 상태: {normalized_status}, 기본값 처리")
            code = -1
        else:
            code = status_map[normalized_status]

        predicted_minutes = model.predict(np.array([[code]]))[0]
        last_time = datetime.fromisoformat(last_time_str)
        arrival_time = last_time + timedelta(minutes=predicted_minutes)
        base_date = arrival_time.date()
        if base_date.weekday() == 6:
            base_date += timedelta(days=1)

        graph_dates = []
        i = -1
        while len(graph_dates) < 5:
            d = base_date + timedelta(days=i)
            graph_dates.append(d)
            i += 1

        weight_map = {
            '집화처리': [0.05, 0.65, 0.20, 0.07, 0.03],
            '간선상차': [0.10, 0.60, 0.15, 0.10, 0.05],
            '간선하차': [0.15, 0.50, 0.20, 0.10, 0.05],
            '배송출발': [0.20, 0.65, 0.10, 0.03, 0.02],
            'sm 입고': [0.07, 0.65, 0.20, 0.05, 0.03]
        }
        default_weights = [0.1, 0.5, 0.2, 0.15, 0.05]
        weights = weight_map.get(normalized_status, default_weights)

        probabilities = [
            0.0 if d.weekday() == 6 else round(weights[i], 4)
            for i, d in enumerate(graph_dates)
        ]

        return jsonify({
            'status': 'success',
            'predicted_minutes': round(predicted_minutes, 1),
            'dates': [d.isoformat() for d in graph_dates],
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/subscribe_alert', methods=['POST'])
def subscribe_alert():
    try:
        data = request.get_json()
        invoice = data.get('invoice')
        user_id = data.get('user_id')
        token = data.get('token')
        carrier_id = data.get('carrier_id')
        status = data.get('status', '')

        print(f"📥 받은 구독 요청 → invoice: {invoice}, status: {data.get('status')}, carrier_id: {carrier_id}, user_id: {user_id}")

        if not invoice or not user_id or not token:
            return jsonify({'status': 'fail', 'message': '필수 항목 누락'}), 400

        for sub in alert_subscriptions:
            if sub['invoice'] == invoice and sub['user_id'] == user_id:
                print(f"⚠️ 중복 등록 시도 감지 → invoice: {invoice}, user_id: {user_id}")
                return jsonify({'status': 'duplicate', 'message': '이미 등록됨'}), 200

        alert_subscriptions.append({
            'invoice': invoice,
            'user_id': user_id,
            'token': token,
            'carrier_id': carrier_id,
            'status': status,
            'subscribed_at': datetime.now().isoformat(),
            'alert_enabled': True
        })

        doc_ref = db.collection("subscriptions").document(f"{user_id}_{invoice}")
        doc_ref.set({
              "invoice": invoice,
              "user_id": user_id,
              "token": token,
              "carrier_id": carrier_id,
              "status": status,
              "subscribed_at": datetime.now().isoformat(),
              "alert_enabled": True
        })


        print(f"✅ 등록 완료 → 현재 구독 수: {len(alert_subscriptions)}")

        return jsonify({'status': 'success', 'message': '알림 등록 완료'}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/unsubscribe_alert', methods=['POST'])
def unsubscribe_alert():
    global alert_subscriptions
    try:
        data = request.get_json()
        invoice = data.get('invoice')
        user_id = data.get('user_id')

        if not invoice or not user_id:
            return jsonify({'status': 'fail', 'message': 'invoice 또는 user_id가 없습니다.'}), 400


        # ✅ Firestore subscriptions 문서 삭제
        doc_ref = db.collection("subscriptions").document(f"{user_id}_{invoice}")
        doc_ref.delete()
        print(f"☁️ Firestore 구독 문서 삭제 완료: {user_id}_{invoice}")

        # ✅ Firestore 메시지도 삭제
        doc_ref = db.collection("messages").document(f"{user_id}_{invoice}")
        doc_ref.delete()
        print(f"☁️ Firestore 메시지 삭제 완료: {user_id}_{invoice}")
        
        load_subscriptions_from_firestore()

        return jsonify({'status': 'success', 'message': '알림 구독 삭제 완료'}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/toggle_alert', methods=['POST'])
def toggle_alert():
    try:
        data = request.get_json()
        invoice = data.get('invoice')
        user_id = data.get('user_id')
        enabled = data.get('enabled', True)

        for sub in alert_subscriptions:
            if sub['invoice'] == invoice and sub['user_id'] == user_id:
                sub['alert_enabled'] = enabled
                save_subscriptions_to_file()

                # ✅ Firestore도 즉시 변경
                doc_ref = db.collection("subscriptions").document(f"{user_id}_{invoice}")
                doc_ref.update({"alert_enabled": enabled})
                print(f"☁️ Firestore alert_enabled 변경 → {user_id}_{invoice}: {enabled}")

                print(f"✔️ toggle_alert 성공 - user_id: {user_id}, invoice: {invoice}, alert_enabled: {enabled}")

                return jsonify({'status': 'success', 'message': '알림 설정 변경됨'})

        return jsonify({'status': 'fail', 'message': '구독 정보 없음'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_current_statuses', methods=['GET'])
def get_current_statuses():
    try:
        subscriptions_ref = db.collection("subscriptions").stream()
        data = [doc.to_dict() for doc in subscriptions_ref]
        return jsonify({'status': 'success', 'subscriptions': data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/alert_messages', methods=['GET'])
def get_alert_messages():
    invoice = request.args.get('invoice')
    user_id = request.args.get('user_id')

    if not invoice or not user_id:
        return jsonify({'status': 'fail', 'message': 'invoice 또는 user_id가 없습니다.'}), 400

    try:
        doc_ref = db.collection("messages").document(f"{user_id}_{invoice}")
        doc = doc_ref.get()
        if doc.exists:
            messages = [msg['body'] for msg in doc.to_dict().get('messages', [])]
        else:
            messages = []
        return jsonify({'status': 'success', 'messages': messages})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


def send_fcm_notification(token, title, body, invoice=None, user_id=None):
    try:
        # FCM 메시지 전송
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token,
            data={}
        )
        response = messaging.send(message)
        print(f"🔔 FCM 전송 성공: {response}")

        # 메시지 Firestore 저장
        if invoice and user_id:
            doc_ref = db.collection("messages").document(f"{user_id}_{invoice}")
            doc = doc_ref.get()
            messages = doc.to_dict().get('messages', []) if doc.exists else []
            messages.append({
                'body': body,
                'timestamp': datetime.now().isoformat()
            })
            doc_ref.set({'messages': messages})
            print(f"☁️ Firestore 메시지 저장 완료 → {user_id}_{invoice}")

    except Exception as e:
        print(f"❗ FCM 전송 실패: {e}")


# 🔍 송장번호로 carrierId 자동 감지 함수
def detect_carrier(tracking_number, access_token):
    query = '''
    query Detect($trackingNumber: String!) {
      detectCarrier(trackingNumber: $trackingNumber) {
        id
        name
      }
    }
    '''
    variables = {"trackingNumber": tracking_number}
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.post(
        'https://apis.tracker.delivery/graphql',
        headers=headers,
        json={'query': query, 'variables': variables}
    )
    result = response.json()
    if 'data' in result and result['data']['detectCarrier']:
        return result['data']['detectCarrier']['id']
    return None

def check_tracking_status():
    """5분마다 실행될 로직"""
    print(f"🧠 PID: {os.getpid()} - 배송 상태 체크 호출")
    load_subscriptions_from_firestore()  # ✅ 최신 데이터를 매번 로드

    try:
        access_token = get_access_token(TRACKER_CLIENT_ID, TRACKER_CLIENT_SECRET)
        print(f"✅ Access Token 생성 성공: {access_token[:10]}...")
    except Exception as e:
        print(f"❗ Access Token 생성 실패: {e}")
        return

    for sub in alert_subscriptions:
        invoice = sub.get('invoice')
        token = sub.get('token')
        user_id = sub.get('user_id')
        prev_status = sub.get('status', '')
        carrier_id = sub.get('carrier_id')

        if not carrier_id:
            print(f"❗ carrierId 없음 - 송장번호: {invoice}")
            continue

        try:
            query = """
            query Track($carrierId: ID!, $trackingNumber: String!) {
              track(carrierId: $carrierId, trackingNumber: $trackingNumber) {
                lastEvent {
                  status {
                    name
                  }
                }
              }
            }
            """
            variables = {"carrierId": carrier_id, "trackingNumber": invoice}
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }
            response = requests.post(
                'https://apis.tracker.delivery/graphql',
                headers=headers,
                json={'query': query, 'variables': variables}
            )
            if response.status_code != 200:
                print(f"❌ [{invoice}] HTTP Status: {response.status_code}")
                continue

            result = response.json()
            if 'errors' in result:
                print(f"❗ [{invoice}] GraphQL 오류 발생: {result['errors']}")
                continue
            if 'data' not in result or not result['data'].get('track'):
                print(f"❗ [{invoice}] 데이터 누락 또는 잘못된 응답.")
                continue

            current_status = result['data']['track']['lastEvent']['status']['name']
            norm_status = normalize_status(current_status)

            if prev_status != norm_status:
                print(f"✅ [{invoice}] 상태 변경 감지: {prev_status} → {norm_status}")

                if prev_status in ['배송완료', '배달완료'] and norm_status in ['배송완료', '배달완료']:
                    print(f"🚫 [{invoice}] 이미 배송완료 상태, 중복 알림 생략")
                    continue

                if sub.get('alert_enabled', True):
                    if norm_status in ['배송완료', '배송 완료', '배달완료', '배달 완료']:
                        try:
                            event_time_str = result['data']['track']['lastEvent']['time']
                            event_time = datetime.fromisoformat(event_time_str)
                            time_str = event_time.strftime("%m월 %d일 %H:%M")
                            message_body = f"{time_str} 배송완료 되었습니다."
                        except Exception as e:
                            print(f"❗ 배송완료 시간 파싱 실패: {e}")
                            message_body = f"배송완료 되었습니다."
                    else:
                        prediction = predict_arrival_internal(current_status, datetime.now().isoformat(), carrier_id)
                        if prediction.get("status") == "success":
                            minutes = prediction["predicted_minutes"]
                            eta = datetime.now() + timedelta(minutes=minutes)
                            eta_str = eta.strftime("%m월 %d일 %H:%M 도착 예상")
                        else:
                            eta_str = "도착 시간 예측 불가"

                        message_body = f"송장번호 : {invoice}\n{norm_status} : {eta_str}"

                    send_fcm_notification(
                        token,
                        "택배 상태 업데이트",
                        message_body,
                        invoice=invoice,
                        user_id=user_id
                    )
                    print(f"🔔 [{invoice}] FCM 알림 전송 완료: {norm_status}")

                else:
                    doc_ref = db.collection("messages").document(f"{user_id}_{invoice}")
                    doc = doc_ref.get()
                    messages = doc.to_dict().get('messages', []) if doc.exists else []
                    messages.append({
                        'body': f"[알림 OFF] 송장번호 : {invoice} 상태변경 : {norm_status}",
                        'timestamp': datetime.now().isoformat()
                    })
                    doc_ref.set({'messages': messages})
                    print(f"☁️ [{invoice}] 메시지만 저장 (알림 OFF) - {norm_status}")

                # ✅ 상태 변경 후 저장
                sub['status'] = norm_status
                save_subscriptions_to_file()
            else:
                print(f"ℹ️ [{invoice}] 상태 변화 없음: {norm_status}")

        except Exception as e:
            print(f"❗ [{invoice}] 예외 발생: {e}")



# ✅ 로드 후 즉시 스케줄러 시작
load_subscriptions_from_file()
load_subscriptions_from_firestore()
print(f"👀 로드된 alert_subscriptions: {alert_subscriptions}")

from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(check_tracking_status, 'interval', minutes=5)
scheduler.start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render가 제공하는 포트 환경변수 사용
    print(f"🚀 서버 시작 - PID: {os.getpid()}, 포트: {port}")
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)