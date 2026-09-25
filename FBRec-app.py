import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 資料庫架構 (Database Schema)
# ==========================================
Base = declarative_base()

class Tipster(Base):
    __tablename__ = 'tipsters'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    tips = relationship("Tip", back_populates="tipster")

class Tip(Base):
    __tablename__ = 'tips'
    id = Column(Integer, primary_key=True)
    tipster_id = Column(Integer, ForeignKey('tipsters.id'))
    category = Column(String(50))      # 賽事分類
    tournament = Column(String(100))   # 賽事名稱
    home_team = Column(String(100))
    away_team = Column(String(100))
    market_type = Column(String(50))   # 盤口類型
    line = Column(Float, default=0.0)  # 盤口線
    selection = Column(String(50))     # 選擇 (主/客/大/小)
    odds = Column(Float, nullable=False)
    
    # 結算狀態
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    corners = Column(Integer, nullable=True)
    status = Column(String(20), default='Open') # Open, Win, Half Win, Push, Half Loss, Loss
    unit_profit = Column(Float, default=0.0)
    
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    tipster = relationship("Tipster", back_populates="tips")

class BankrollLedger(Base):
    __tablename__ = 'bankroll_ledger'
    id = Column(Integer, primary_key=True)
    account_type = Column(String(20)) # 'User' or 'Tipster'
    tipster_id = Column(Integer, ForeignKey('tipsters.id'), nullable=True)
    amount = Column(Float, nullable=False) # 正為存/贏, 負為提/輸
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.now)

class ActionLog(Base):
    __tablename__ = 'action_logs'
    id = Column(Integer, primary_key=True)
    action_type = Column(String(50)) # 'DELETE', 'SETTLE'
    record_id = Column(Integer)
    table_name = Column(String(50))
    old_data = Column(Text) # JSON format for rollback
    created_at = Column(DateTime, default=datetime.now)

# 初始化資料庫
engine = create_engine('sqlite:///betting_system.db', echo=False, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ==========================================
# 2. 核心算法與結算引擎 (Settlement & Odds)
# ==========================================
def calculate_linked_odds(input_odds, margin=1.085):
    """根據輸入賠率和抽水，反推對家賠率"""
    if input_odds <= 1.0: return 0.0
    prob_1 = 1 / input_odds
    prob_2 = margin - prob_1
    if prob_2 <= 0: return 0.0
    return round(1 / prob_2, 3)

def calculate_actual_margin(odds_1, odds_2):
    """手動覆寫時，反推當前實際抽水"""
    if odds_1 <= 1 or odds_2 <= 1: return 0.0
    return round((1/odds_1) + (1/odds_2), 4)

def settle_asian_handicap(market_type, line, selection, home_score, away_score, corners_total=None):
    """亞洲盤口結算引擎 (返回: status, unit_profit)"""
    if '大小' in market_type or ('角' in market_type and '讓' not in market_type):
        actual_total = corners_total if '角' in market_type else (home_score + away_score)
        diff = actual_total - line
        is_over = (selection == '大')
        net_diff = diff if is_over else -diff
    else:
        diff = (home_score - away_score) + line
        is_home = (selection == '主')
        net_diff = diff if is_home else -diff

    if net_diff > 0.25: return 'Win', 1.0
    elif net_diff == 0.25: return 'Half Win', 0.5
    elif net_diff == 0.0: return 'Push', 0.0
    elif net_diff == -0.25: return 'Half Loss', -0.5
    else: return 'Loss', -1.0

# ==========================================
# 3. 機器學習與凱利精算模型 (ML & Kelly)
# ==========================================
def get_tipster_features(tipster_id, session):
    tips = session.query(Tip).filter(
        Tip.tipster_id == tipster_id, 
        Tip.status != 'Open',
        Tip.is_deleted == False
    ).all()

    if len(tips) < 10: return None, None

    data = []
    for t in tips:
        target = 1 if t.status in ['Win', 'Half Win'] else 0
        data.append({'odds': t.odds, 'line': t.line, 'target': target})
        
    df = pd.DataFrame(data)
    X = df[['odds', 'line']]
    y = df['target']
    return X, y

def predict_win_prob_and_kelly(tipster_id, current_odds, current_line, user_bankroll, session, kelly_fraction=0.25):
    X, y = get_tipster_features(tipster_id, session)
    
    if X is None or len(X) < 10:
        p_win = 0.5
        ai_status = "數據不足，採用保守估計 (最少需 10 筆已結算紀錄)"
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        p_win = model.predict_proba([[current_odds, current_line]])[0][1]
        ai_status = "RF 模型運算中"

    b = current_odds - 1.0
    if b <= 0: return 0.0, p_win, "賠率異常"
    
    q = 1.0 - p_win
    kelly_f = (b * p_win - q) / b
    kelly_f = max(0, kelly_f) * kelly_fraction
    suggested_stake = max(0, user_bankroll * kelly_f)
    
    return round(suggested_stake, 2), round(p_win, 4), ai_status

# ==========================================
# 4. 彈出視窗與資料庫管理 (Dialogs)
# ==========================================
@st.dialog("📊 數據庫即時預覽與管理", width="large")
def preview_db_dialog():
    session = SessionLocal()
    tips = session.query(Tip).filter(Tip.is_deleted == False).order_by(Tip.id.desc()).all()
    
    if not tips:
        st.info("目前無任何資料")
        session.close()
        return

    df = pd.DataFrame([{
        'ID': t.id,
        '賽事': t.tournament,
        '對陣': f"{t.home_team} vs {t.away_team}",
        '盤口': f"{t.market_type} ({t.line})",
        '賠率': t.odds,
        '狀態': t.status,
        '盈虧單位': t.unit_profit
    } for t in tips])

    search_kw = st.text_input("🔍 關鍵字搜尋 (賽事、球隊):")
    if search_kw:
        df = df[df.apply(lambda row: row.astype(str).str.contains(search_kw).any(), axis=1)]
    
    st.dataframe(df, use_container_width=True)
    st.markdown(f"**匯總**：總單數 **{len(df)}** | 總盈虧單位 **{df['盈虧單位'].sum():.2f}**")

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="📥 匯出為 CSV", data=csv, file_name='betting_history.csv', mime='text/csv')

    st.divider()
    st.subheader("⏪ 撤銷與回滾中心 (Undo Stack)")
    logs = session.query(ActionLog).order_by(ActionLog.id.desc()).limit(10).all()
    if logs:
        for log in logs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"時間: {log.created_at.strftime('%m-%d %H:%M')} | 動作: {log.action_type} | 紀錄 ID: {log.record_id}")
            if col2.button("復原", key=f"undo_{log.id}"):
                target_tip = session.query(Tip).get(log.record_id)
                if target_tip:
                    old_data = json.loads(log.old_data)
                    target_tip.status = old_data.get('status', 'Open')
                    target_tip.is_deleted = old_data.get('is_deleted', False)
                    target_tip.unit_profit = old_data.get('unit_profit', 0.0)
                    target_tip.home_score = old_data.get('home_score', None)
                    target_tip.away_score = old_data.get('away_score', None)
                    target_tip.corners = old_data.get('corners', None)
                    session.delete(log)
                    session.commit()
                    st.rerun()
    else:
        st.write("近期無可復原的操作。")
    session.close()

# ==========================================
# 5. 主程式與 UI 渲染 (Main Streamlit App)
# ==========================================
def main():
    st.set_page_config(page_title="Tipster Quant AI", layout="wide")

    # Session State 初始化
    if 'odds1' not in st.session_state: st.session_state.odds1 = 1.85
    if 'odds2' not in st.session_state: st.session_state.odds2 = calculate_linked_odds(1.85)

    def update_odds1():
        st.session_state.odds2 = calculate_linked_odds(st.session_state.odds1)

    def update_odds2():
        st.session_state.odds1 = calculate_linked_odds(st.session_state.odds2)

    db_session = SessionLocal()
    tipsters = db_session.query(Tipster).all()
    tipster_names = [t.name for t in tipsters]

    # 建立動態 Tabs
    tabs = st.tabs(tipster_names + ["➕ 新增分享者", "💰 資金與 AI 總覽", "⚙️ 系統管理"])

    # ----------------------------------------
    # [Tab] 新增分享者
    # ----------------------------------------
    with tabs[-3]:
        st.subheader("新增分享者")
        new_name = st.text_input("分享者稱號")
        if st.button("創建分享者"):
            if new_name and new_name not in tipster_names:
                db_session.add(Tipster(name=new_name))
                db_session.commit()
                st.success("創建成功！")
                st.rerun()
            else:
                st.error("名稱不能為空或已存在")

    # ----------------------------------------
    # [Tab] 系統管理
    # ----------------------------------------
    with tabs[-1]:
        st.subheader("系統管理員工具")
        st.button("🗄️ 開啟數據庫管理 (預覽 / Undo / 匯出)", on_click=preview_db_dialog)

    # ----------------------------------------
    # [Tab] 動態 Tipster Tabs
    # ----------------------------------------
    for i, tipster in enumerate(tipsters):
        with tabs[i]:
            st.header(f"分享者：{tipster.name} 的盤口建議")
            colL, colR = st.columns([2, 1])
            
            with colL:
                st.subheader("📝 紀錄新建議")
                cat = st.selectbox("賽事分類", ["國內聯賽", "國內盃賽", "國際聯賽", "國際盃賽", "友誼賽"], key=f"cat_{tipster.id}")
                
                # 自動補全選項提取
                all_tips = db_session.query(Tip).all()
                history_teams = list(set([t.home_team for t in all_tips] + [t.away_team for t in all_tips]))
                history_tours = list(set([t.tournament for t in all_tips if t.tournament]))
                
                tour = st.selectbox("賽事名稱 (選擇或手動輸入)", [""] + history_tours, key=f"tour_{tipster.id}")
                if not tour: tour = st.text_input("輸入新賽事名稱", key=f"tour_new_{tipster.id}")
                
                c1, c2 = st.columns(2)
                with c1: 
                    h_team = st.selectbox("主隊名稱", [""] + history_teams, key=f"ht_{tipster.id}")
                    if not h_team: h_team = st.text_input("輸入新主隊", key=f"ht_new_{tipster.id}")
                with c2: 
                    a_team = st.selectbox("客隊名稱", [""] + history_teams, key=f"at_{tipster.id}")
                    if not a_team: a_team = st.text_input("輸入新客隊", key=f"at_new_{tipster.id}")

                market = st.selectbox("盤口", ["讓球", "半場讓球", "入球大小", "角球大小", "半場角球大小", "讓角", "半場讓角"], key=f"mk_{tipster.id}")
                
                default_line = 2.5 if "入球" in market else (9.5 if market=="角球大小" else (4.5 if market=="半場角球大小" else 0.0))
                line = st.number_input("盤口線", value=default_line, step=0.25, key=f"line_{tipster.id}")
                
                selection = st.radio("您的選擇", ["主", "客", "大", "小"], horizontal=True, key=f"sel_{tipster.id}")
                
                c_o1, c_o2, c_o3 = st.columns(3)
                with c_o1:
                    st.number_input("賠率 (主/大)", min_value=1.01, step=0.01, key="odds1", on_change=update_odds1)
                with c_o2:
                    st.number_input("賠率 (客/小)", min_value=1.01, step=0.01, key="odds2", on_change=update_odds2)
                with c_o3:
                    margin = calculate_actual_margin(st.session_state.odds1, st.session_state.odds2)
                    st.info(f"當前抽水: \n**{margin:.4f}**")

                if st.button("提交建議", key=f"submit_{tipster.id}"):
                    if not tour or not h_team or not a_team:
                        st.error("賽事、主隊、客隊名稱不能為空")
                    else:
                        new_tip = Tip(
                            tipster_id=tipster.id, category=cat, tournament=tour,
                            home_team=h_team, away_team=a_team, market_type=market,
                            line=line, selection=selection, odds=st.session_state.odds1
                        )
                        db_session.add(new_tip)
                        db_session.commit()
                        st.success("紀錄成功！")
                        st.rerun()

            with colR:
                st.subheader("🤖 AI 精算模型")
                settled_count = db_session.query(Tip).filter(Tip.tipster_id == tipster.id, Tip.status != 'Open').count()
                st.metric("該分享者已結算數據筆數", settled_count)
                
                user_ledger = db_session.query(BankrollLedger).filter_by(account_type='User').all()
                user_bankroll = sum([l.amount for l in user_ledger]) if user_ledger else 10000.0
                
                rec_stake, win_prob, ai_status = predict_win_prob_and_kelly(
                    tipster.id, st.session_state.odds1, line, user_bankroll, db_session
                )
                
                st.caption(f"模型狀態: {ai_status}")
                st.metric("預測勝率 (RF Model)", f"{win_prob * 100:.1f}%")
                st.metric("建議下注額 (Kelly)", f"${rec_stake}")

    # ----------------------------------------
    # [Tab] 資金與 AI 總覽 (t_settle)
    # ----------------------------------------
    with tabs[-2]:
        st.header("資金管理與賽果結算")
        c_fund1, c_fund2 = st.columns(2)
        with c_fund1:
            st.subheader("注入/提取真實資金")
            amt = st.number_input("金額 (正為存, 負為取)", step=100.0)
            desc = st.text_input("備註")
            if st.button("提交資金變動"):
                db_session.add(BankrollLedger(account_type='User', amount=amt, description=desc))
                db_session.commit()
                st.success("資金流水已紀錄")
        
        with c_fund2:
            user_ledger = db_session.query(BankrollLedger).filter_by(account_type='User').all()
            current_br = sum([l.amount for l in user_ledger])
            st.metric("📊 用家真實本金總額", f"${current_br:.2f}")

        st.divider()
        st.subheader("📝 待結算注單 (t_settle)")
        open_tips = db_session.query(Tip).filter_by(status='Open', is_deleted=False).all()
        
        if not open_tips:
            st.info("目前無待結算注單。")
            
        for t in open_tips:
            with st.expander(f"ID:{t.id} | {t.tournament} | {t.home_team} vs {t.away_team} | {t.market_type} ({t.line}) - 賠率 {t.odds}"):
                c1, c2, c3, c4 = st.columns(4)
                h_sc = c1.number_input("主隊進球", min_value=0, step=1, key=f"h_sc_{t.id}")
                a_sc = c2.number_input("客隊進球", min_value=0, step=1, key=f"a_sc_{t.id}")
                cor = c3.number_input("角球數", min_value=0, step=1, key=f"cor_{t.id}")
                if c4.button("結算", key=f"set_{t.id}"):
                    # 紀錄 Undo 狀態
                    old_data = {
                        'status': t.status, 'is_deleted': t.is_deleted, 
                        'unit_profit': t.unit_profit, 'home_score': t.home_score,
                        'away_score': t.away_score, 'corners': t.corners
                    }
                    db_session.add(ActionLog(
                        action_type='SETTLE', record_id=t.id, 
                        table_name='tips', old_data=json.dumps(old_data)
                    ))

                    # 結算引擎計算
                    status, profit = settle_asian_handicap(t.market_type, t.line, t.selection, h_sc, a_sc, cor)
                    t.home_score, t.away_score, t.corners = h_sc, a_sc, cor
                    t.status = status
                    t.unit_profit = profit
                    
                    db_session.add(BankrollLedger(
                        account_type='Tipster', tipster_id=t.tipster_id, 
                        amount=profit * 100, # 假設每單位為 $100
                        description=f"結算單 {t.id} - {status}"
                    ))
                    db_session.commit()
                    st.success(f"結算完成：{status}, 獲利單位：{profit}")
                    st.rerun()

    db_session.close()

if __name__ == "__main__":
    main()
