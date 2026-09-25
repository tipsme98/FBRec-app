import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, inspect, text
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
    category = Column(String(50))      
    tournament = Column(String(100))   
    home_team = Column(String(100))
    away_team = Column(String(100))
    market_type = Column(String(50))   
    line = Column(Float, default=0.0)  
    selection = Column(String(50))     
    odds = Column(Float, nullable=False)
    stake = Column(Float, default=0.0)    
    
    # 結算狀態 (新增半場與全場區分)
    home_score = Column(Integer, nullable=True)         
    away_score = Column(Integer, nullable=True)         
    home_ht_score = Column(Integer, nullable=True)      
    away_ht_score = Column(Integer, nullable=True)      
    home_corners = Column(Integer, nullable=True)       
    away_corners = Column(Integer, nullable=True)       
    home_ht_corners = Column(Integer, nullable=True)    
    away_ht_corners = Column(Integer, nullable=True)    
    
    status = Column(String(20), default='Open') 
    unit_profit = Column(Float, default=0.0) 
    profit = Column(Float, default=0.0)      
    payout = Column(Float, default=0.0)      
    
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    tipster = relationship("Tipster", back_populates="tips")

class BankrollLedger(Base):
    __tablename__ = 'bankroll_ledger'
    id = Column(Integer, primary_key=True)
    account_type = Column(String(20)) 
    tipster_id = Column(Integer, ForeignKey('tipsters.id'), nullable=True)
    amount = Column(Float, nullable=False) 
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.now)

class ActionLog(Base):
    __tablename__ = 'action_logs'
    id = Column(Integer, primary_key=True)
    action_type = Column(String(50)) 
    record_id = Column(Integer)
    ledger_id = Column(Integer, nullable=True) 
    table_name = Column(String(50))
    old_data = Column(Text) 
    created_at = Column(DateTime, default=datetime.now)

# 初始化資料庫
engine = create_engine('sqlite:///betting_system.db', echo=False, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)

# 自動升級舊有資料庫 (Auto-Migration)
def upgrade_database(engine):
    inspector = inspect(engine)
    with engine.begin() as conn:
        if 'tips' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('tips')]
            if 'stake' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN stake FLOAT DEFAULT 0.0"))
            if 'profit' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN profit FLOAT DEFAULT 0.0"))
            if 'payout' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN payout FLOAT DEFAULT 0.0"))
            if 'home_corners' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN home_corners INTEGER"))
            if 'away_corners' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN away_corners INTEGER"))
            if 'home_ht_score' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN home_ht_score INTEGER"))
            if 'away_ht_score' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN away_ht_score INTEGER"))
            if 'home_ht_corners' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN home_ht_corners INTEGER"))
            if 'away_ht_corners' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN away_ht_corners INTEGER"))

upgrade_database(engine)
SessionLocal = sessionmaker(bind=engine)

# ==========================================
# 2. 核心算法與結算引擎
# ==========================================
def calculate_linked_odds(input_odds, margin=1.085):
    if input_odds <= 1.0: return 0.0
    prob_1 = 1 / input_odds
    prob_2 = margin - prob_1
    if prob_2 <= 0: return 0.0
    return round(1 / prob_2, 3)

def calculate_actual_margin(odds_1, odds_2):
    if odds_1 <= 1 or odds_2 <= 1: return 0.0
    return (1/odds_1) + (1/odds_2)

def settle_asian_handicap(market_type, line, selection, 
                          h_sc, a_sc, h_ht_sc, a_ht_sc, 
                          h_cor, a_cor, h_ht_cor, a_ht_cor, odds=1.85):
    # 決定使用的核心數據
    if market_type == "讓球":
        h_val, a_val = h_sc, a_sc
    elif market_type == "半場讓球":
        h_val, a_val = h_ht_sc, a_ht_sc
    elif market_type == "讓角":
        h_val, a_val = h_cor, a_cor
    elif market_type == "半場讓角":
        h_val, a_val = h_ht_cor, a_ht_cor
    elif market_type == "入球大小":
        total = h_sc + a_sc
    elif market_type == "半場入球大小":
        total = h_ht_sc + a_ht_sc
    elif market_type == "角球大小":
        total = h_cor + a_cor
    elif market_type == "半場角球大小":
        total = h_ht_cor + a_ht_cor

    # 計算差值
    if '大小' in market_type:
        diff = total - line
        is_over = (selection == '大')
        net_diff = diff if is_over else -diff
    else:
        diff = (h_val - a_val) + line
        is_home = (selection == '主')
        net_diff = diff if is_home else -diff

    if net_diff > 0.25: return 'Win', round(odds - 1.0, 3)
    elif net_diff == 0.25: return 'Half Win', round((odds - 1.0) / 2, 3)
    elif net_diff == 0.0: return 'Push', 0.0
    elif net_diff == -0.25: return 'Half Loss', -0.5
    else: return 'Loss', -1.0

# ==========================================
# 3. 機器學習與凱利精算模型
# ==========================================
def get_tipster_features(tipster_id, session):
    tips = session.query(Tip).filter(
        Tip.tipster_id == tipster_id, 
        Tip.status != 'Open',
        Tip.is_deleted == False
    ).all()

    if len(tips) < 10: return None, None
    data = [{'odds': t.odds, 'line': t.line, 'target': 1 if t.status in ['Win', 'Half Win'] else 0} for t in tips]
    df = pd.DataFrame(data)
    return df[['odds', 'line']], df['target']

def predict_win_prob_and_kelly(tipster_id, current_odds, current_line, market_type, user_bankroll, session, kelly_fraction=0.25):
    X, y = get_tipster_features(tipster_id, session)
    
    if X is None or len(X) < 10:
        p_win = 0.5
        ai_status = "數據不足，採用保守估計"
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        p_win = model.predict_proba([[current_odds, current_line]])[0][1]
        ai_status = "RF 模型運算中"

    b = current_odds - 1.0
    if b <= 0: return 0.0, p_win, "賠率異常"
    
    q = 1.0 - p_win
    kelly_f = max(0, (b * p_win - q) / b) * kelly_fraction
    suggested_stake = max(0, user_bankroll * kelly_f)
    
    # 資金風控與上下限邏輯
    max_stake = user_bankroll * 0.10 # 最高限制 10%
    
    if suggested_stake > 0:
        if market_type == "讓球":
            if suggested_stake < 200:
                suggested_stake = 200 if p_win >= 0.50 else 0
            # 確保不會因為強制最低額度而打破 10% 限制
            if suggested_stake > max_stake:
                suggested_stake = max_stake if max_stake >= 200 else 0
        else:
            if suggested_stake < 10:
                suggested_stake = 10 if p_win >= 0.50 else 0
            if suggested_stake > max_stake:
                suggested_stake = max_stake if max_stake >= 10 else 0

    return round(suggested_stake, 2), round(p_win, 4), ai_status

# ==========================================
# 4. 財務統整與 UI 視覺函數
# ==========================================
def style_financials(val):
    if isinstance(val, (int, float)):
        color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
        return f'color: {color}; font-weight: bold;'
    return ''

def get_bankroll_summary(session):
    all_ledgers = session.query(BankrollLedger).all()
    all_settled_tips = session.query(Tip).filter(Tip.status != 'Open', Tip.is_deleted == False).all()
    data = []
    
    # 用家 (User) 核算
    u_ledgers = [l for l in all_ledgers if l.account_type == 'User']
    u_dep = sum(l.amount for l in u_ledgers if l.amount < 0)
    u_wit = sum(l.amount for l in u_ledgers if l.amount > 0)
    u_profit = sum(t.profit for t in all_settled_tips) 
    u_avail = abs(u_dep) - u_wit + u_profit
    
    data.append({
        '帳戶名稱': '👑 用家 (User)',
        '總存入本金': u_dep,
        '總提取本金': u_wit,
        '累計總盈虧': u_profit,
        '當前可用資金': u_avail
    })
    
    # 各分享者 (Tipster) 核算
    tipsters = session.query(Tipster).all()
    for tipster in tipsters:
        t_ledgers = [l for l in all_ledgers if l.account_type == 'Tipster' and l.tipster_id == tipster.id]
        t_dep = sum(l.amount for l in t_ledgers if l.amount < 0)
        t_wit = sum(l.amount for l in t_ledgers if l.amount > 0)
        t_profit = sum(t.profit for t in all_settled_tips if t.tipster_id == tipster.id)
        t_avail = abs(t_dep) - t_wit + t_profit
        
        data.append({
            '帳戶名稱': f'👤 {tipster.name}',
            '總存入本金': t_dep,
            '總提取本金': t_wit,
            '累計總盈虧': t_profit,
            '當前可用資金': t_avail
        })
        
    return pd.DataFrame(data), u_avail

# ==========================================
# 5. 彈出視窗與資料庫管理 (Dialogs)
# ==========================================
@st.dialog("📊 數據庫即時預覽與管理", width="large")
def preview_db_dialog():
    session = SessionLocal()
    tipsters = session.query(Tipster).all()
    
    if not tipsters:
        st.info("目前無任何資料")
        session.close()
        return

    tabs = st.tabs([t.name for t in tipsters] + ["💰 用家資金流水"])
    
    for i, tipster in enumerate(tipsters):
        with tabs[i]:
            tips = session.query(Tip).filter(Tip.tipster_id == tipster.id, Tip.is_deleted == False).order_by(Tip.id.desc()).all()
            if not tips:
                st.info(f"{tipster.name} 尚無投注紀錄")
                continue

            data = []
            for t in tips:
                res = f"半場 {t.home_ht_score}-{t.away_ht_score} | 全場 {t.home_score}-{t.away_score}" if t.status != 'Open' else "未結算"
                data.append({
                    'ID': t.id,
                    '賽事': t.tournament,
                    '對陣': f"{t.home_team} vs {t.away_team}",
                    '盤口': f"{t.market_type} ({t.line})",
                    '賠率': t.odds,
                    '投注額': t.stake,
                    '比賽結果': res,
                    '狀態': t.status,
                    '單位盈虧': t.unit_profit,
                    '盈虧': t.profit,
                    '派彩': t.payout
                })
            
            df = pd.DataFrame(data)
            float_cols = ['賠率', '投注額', '單位盈虧', '盈虧', '派彩']
            for c in float_cols: df[c] = df[c].astype(float)
            
            st.dataframe(df.style.map(style_financials, subset=['單位盈虧', '盈虧', '派彩']).format({col: "{:.2f}" for col in float_cols}), use_container_width=True)
            st.markdown(f"**匯總**：總單數 **{len(df)}** | 總投注額 **${df['投注額'].sum():.2f}** | 總盈虧 **${df['盈虧'].sum():.2f}** | 總單位盈虧 **{df['單位盈虧'].sum():.2f}**")
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label=f"📥 匯出 {tipster.name} 報表", data=csv, file_name=f'{tipster.name}_history.csv', mime='text/csv', key=f"dl_{tipster.id}")

    with tabs[-1]:
        ledgers = session.query(BankrollLedger).filter_by(account_type='User').order_by(BankrollLedger.id.desc()).all()
        if ledgers:
            ldf = pd.DataFrame([{
                '時間': l.created_at.strftime('%Y-%m-%d %H:%M'),
                '類型': '存入(負)' if l.amount < 0 else '提取(正)',
                '金額': l.amount,
                '備註': l.description
            } for l in ledgers])
            ldf['金額'] = ldf['金額'].astype(float)
            st.dataframe(ldf.style.map(style_financials, subset=['金額']).format({'金額': "{:.2f}"}), use_container_width=True)
        else:
            st.info("用家尚無存取資金流水")

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
                    target_tip.profit = old_data.get('profit', 0.0)
                    target_tip.payout = old_data.get('payout', 0.0)
                    
                    target_tip.home_score = old_data.get('home_score')
                    target_tip.away_score = old_data.get('away_score')
                    target_tip.home_ht_score = old_data.get('home_ht_score')
                    target_tip.away_ht_score = old_data.get('away_ht_score')
                    target_tip.home_corners = old_data.get('home_corners')
                    target_tip.away_corners = old_data.get('away_corners')
                    target_tip.home_ht_corners = old_data.get('home_ht_corners')
                    target_tip.away_ht_corners = old_data.get('away_ht_corners')
                        
                    session.delete(log)
                    session.commit()
                    st.rerun()
    else:
        st.write("近期無可復原的操作。")
    session.close()

# ==========================================
# 6. 主程式與 UI 渲染 (Main App)
# ==========================================
def main():
    st.set_page_config(page_title="Tipster Quant AI", layout="wide")

    if 'odds1' not in st.session_state: st.session_state.odds1 = 1.85
    if 'odds2' not in st.session_state: st.session_state.odds2 = calculate_linked_odds(1.85)

    def update_odds1():
        st.session_state.odds2 = calculate_linked_odds(st.session_state.odds1)

    db_session = SessionLocal()
    tipsters = db_session.query(Tipster).all()
    tipster_names = [t.name for t in tipsters]
    tipster_dict = {t.name: t.id for t in tipsters}

    tabs = st.tabs(tipster_names + ["➕ 新增分享者", "💰 資金總覽與結算", "⚙️ 系統管理"])

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
        st.button("🗄️ 開啟數據庫管理 (預覽 / 撤銷 / 匯出)", on_click=preview_db_dialog)

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
                
                all_tips = db_session.query(Tip).all()
                history_teams = list(set([t.home_team for t in all_tips] + [t.away_team for t in all_tips]))
                history_tours = list(set([t.tournament for t in all_tips if t.tournament]))
                
                tour = st.selectbox("賽事名稱", [""] + history_tours, key=f"tour_{tipster.id}")
                if not tour: tour = st.text_input("或輸入新賽事名稱", key=f"tour_new_{tipster.id}")
                
                c1, c2 = st.columns(2)
                with c1: 
                    h_team = st.selectbox("主隊名稱", [""] + history_teams, key=f"ht_{tipster.id}")
                    if not h_team: h_team = st.text_input("或輸入新主隊", key=f"ht_new_{tipster.id}")
                with c2: 
                    a_team = st.selectbox("客隊名稱", [""] + history_teams, key=f"at_{tipster.id}")
                    if not a_team: a_team = st.text_input("或輸入新客隊", key=f"at_new_{tipster.id}")

                market = st.selectbox("盤口", ["讓球", "半場讓球", "入球大小", "半場入球大小", "角球大小", "半場角球大小", "讓角", "半場讓角"], key=f"mk_{tipster.id}")
                
                # 【重要修正】利用盤口名稱動態賦值預設盤口線
                if market in ["讓球", "半場讓球", "讓角", "半場讓角"]: default_line = 0.0
                elif market == "入球大小": default_line = 2.5
                elif market == "半場入球大小": default_line = 1.5
                elif market == "角球大小": default_line = 9.5
                elif market == "半場角球大小": default_line = 4.5
                else: default_line = 0.0
                
                # 【重要修正】動態改變 key，強制 Streamlit 切換盤口時載入 default_line
                line = st.number_input("盤口線", value=default_line, step=0.25, key=f"line_{tipster.id}_{market}")
                selection = st.radio("您的選擇", ["主", "客", "大", "小"], horizontal=True, key=f"sel_{tipster.id}")
                
                c_o1, c_o2, c_o3 = st.columns(3)
                with c_o1:
                    st.number_input("賠率 (主/大)", min_value=1.01, step=0.01, key="odds1", on_change=update_odds1)
                with c_o2:
                    st.number_input("賠率 (客/小)", min_value=1.01, step=0.01, key="odds2")
                with c_o3:
                    margin = calculate_actual_margin(st.session_state.odds1, st.session_state.odds2)
                    margin_pct = (margin - 1.0) * 100 if margin > 1 else 0
                    st.info(f"抽水: **{margin:.4f}**\n\n({margin_pct:.2f}%)")

            with colR:
                st.subheader("🤖 AI 預測與凱利精算")
                _, user_avail_bankroll = get_bankroll_summary(db_session)
                
                rec_stake, win_prob, ai_status = predict_win_prob_and_kelly(
                    tipster.id, st.session_state.odds1, line, market, user_avail_bankroll, db_session
                )
                
                st.caption(f"模型狀態: {ai_status}")
                st.metric("預測勝率 (RF Model)", f"{win_prob * 100:.1f}%")
                st.metric("建議下注額 (Kelly, 最高不超過本金 10%)", f"${rec_stake:.2f}")
                
            with colL:
                stake = st.number_input("實際下注額 (預設為AI建議)", value=float(rec_stake), step=10.0, key=f"stake_{tipster.id}")
                
                if st.button("提交建議與注單", key=f"submit_{tipster.id}", type="primary"):
                    if not tour or not h_team or not a_team:
                        st.error("賽事、主隊、客隊名稱不能為空")
                    else:
                        new_tip = Tip(
                            tipster_id=tipster.id, category=cat, tournament=tour,
                            home_team=h_team, away_team=a_team, market_type=market,
                            line=line, selection=selection, odds=st.session_state.odds1, stake=stake
                        )
                        db_session.add(new_tip)
                        db_session.commit()
                        st.success("紀錄成功！")
                        st.rerun()

    # ----------------------------------------
    # [Tab] 資金總覽與結算 (t_settle)
    # ----------------------------------------
    with tabs[-2]:
        st.header("資金總覽與賽果結算")
        
        # --- 存取資金操作 ---
        with st.expander("📝 存取資金操作 (注入/提取本金)"):
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                acc_type = st.radio("對象", ["用家真實資金", "分享者虛擬本金"], horizontal=True)
                target_tipster = st.selectbox("選擇分享者", tipster_names) if acc_type == "分享者虛擬本金" else None
            with c_f2:    
                action = st.radio("操作", ["存入 (紅色負數：投入本金)", "提取 (綠色正數：回收本金)"], horizontal=True)
                amt_input = st.number_input("金額", min_value=0.0, step=100.0)
                desc = st.text_input("備註 (選填)")
            
            if st.button("提交資金變動"):
                if amt_input > 0:
                    actual_amt = round(-amt_input if "存入" in action else amt_input, 2)
                    db_acc = 'Tipster' if acc_type == "分享者虛擬本金" else 'User'
                    t_id = tipster_dict.get(target_tipster) if target_tipster else None
                    
                    db_session.add(BankrollLedger(account_type=db_acc, tipster_id=t_id, amount=actual_amt, description=desc))
                    db_session.commit()
                    st.success("資金流水已成功紀錄！")
                    st.rerun()

        # --- 表列資金總覽 ---
        st.subheader("📊 資金總覽 (表列方式)")
        df_summary, _ = get_bankroll_summary(db_session)
        df_summary_cols = ['總存入本金', '總提取本金', '累計總盈虧', '當前可用資金']
        for c in df_summary_cols: df_summary[c] = df_summary[c].astype(float)
        
        st.dataframe(df_summary.style.map(style_financials, subset=df_summary_cols).format({c: "{:.2f}" for c in df_summary_cols}), use_container_width=True)
        st.divider()

        # --- 賽果結算區 (8大維度) ---
        st.subheader("🏁 待結算注單")
        open_tips = db_session.query(Tip).filter_by(status='Open', is_deleted=False).all()
        
        if not open_tips:
            st.info("目前無待結算注單。")
            
        for t in open_tips:
            with st.expander(f"ID:{t.id} | {t.tournament} | {t.home_team} vs {t.away_team} | {t.market_type} ({t.line}) | 投注額: ${t.stake:.2f}"):
                
                # 8大輸入欄
                c1, c2, c3, c4 = st.columns(4)
                h_ht_sc = c1.number_input("主隊半場進球", min_value=0, step=1, key=f"h_ht_sc_{t.id}")
                a_ht_sc = c2.number_input("客隊半場進球", min_value=0, step=1, key=f"a_ht_sc_{t.id}")
                h_sc = c3.number_input("主隊全場進球", min_value=0, step=1, key=f"h_sc_{t.id}")
                a_sc = c4.number_input("客隊全場進球", min_value=0, step=1, key=f"a_sc_{t.id}")

                c5, c6, c7, c8 = st.columns(4)
                h_ht_cor = c5.number_input("主隊半場角球", min_value=0, step=1, key=f"h_ht_cor_{t.id}")
                a_ht_cor = c6.number_input("客隊半場角球", min_value=0, step=1, key=f"a_ht_cor_{t.id}")
                h_cor = c7.number_input("主隊全場角球", min_value=0, step=1, key=f"h_cor_{t.id}")
                a_cor = c8.number_input("客隊全場角球", min_value=0, step=1, key=f"a_cor_{t.id}")
                
                # 系統動態計算與顯示
                st.info(f"📊 **系統計算**：兩隊半場總進球 {h_ht_sc + a_ht_sc} | 全場總進球 {h_sc + a_sc} | 半場總角球 {h_ht_cor + a_ht_cor} | 全場總角球 {h_cor + a_cor}")
                
                if st.button("執行結算", key=f"set_{t.id}", type="primary"):
                    status, unit_profit = settle_asian_handicap(
                        t.market_type, t.line, t.selection, 
                        h_sc, a_sc, h_ht_sc, a_ht_sc, 
                        h_cor, a_cor, h_ht_cor, a_ht_cor, t.odds
                    )
                    
                    profit = round(t.stake * unit_profit, 2)
                    payout = round(t.stake + profit if unit_profit >= -0.5 else 0.0, 2)
                    
                    # 紀錄 Undo 狀態
                    old_data = {
                        'status': t.status, 'is_deleted': t.is_deleted, 
                        'unit_profit': t.unit_profit, 'profit': t.profit, 'payout': t.payout,
                        'home_score': t.home_score, 'away_score': t.away_score,
                        'home_ht_score': t.home_ht_score, 'away_ht_score': t.away_ht_score,
                        'home_corners': t.home_corners, 'away_corners': t.away_corners,
                        'home_ht_corners': t.home_ht_corners, 'away_ht_corners': t.away_ht_corners
                    }
                    db_session.add(ActionLog(
                        action_type='SETTLE', record_id=t.id, 
                        table_name='tips', old_data=json.dumps(old_data)
                    ))

                    # 更新寫入結果
                    t.home_score, t.away_score = h_sc, a_sc
                    t.home_ht_score, t.away_ht_score = h_ht_sc, a_ht_sc
                    t.home_corners, t.away_corners = h_cor, a_cor
                    t.home_ht_corners, t.away_ht_corners = h_ht_cor, a_ht_cor
                    
                    t.status = status
                    t.unit_profit = unit_profit
                    t.profit = profit
                    t.payout = payout
                    
                    db_session.commit()
                    st.success(f"結算完成：{status} | 單位盈虧：{unit_profit:.2f} | 實際盈虧：${profit:.2f}")
                    st.rerun()

    db_session.close()

if __name__ == "__main__":
    main()
