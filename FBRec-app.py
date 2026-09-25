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
    
    # 結算狀態
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    home_corners = Column(Integer, nullable=True) 
    away_corners = Column(Integer, nullable=True) 
    
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

# 自動升級舊有資料庫 (解決 OperationalError: no such column)
def upgrade_database(engine):
    inspector = inspect(engine)
    with engine.begin() as conn:
        if 'tips' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('tips')]
            if 'stake' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN stake FLOAT DEFAULT 0.0"))
            if 'home_corners' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN home_corners INTEGER"))
            if 'away_corners' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN away_corners INTEGER"))
            if 'profit' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN profit FLOAT DEFAULT 0.0"))
            if 'payout' not in columns: conn.execute(text("ALTER TABLE tips ADD COLUMN payout FLOAT DEFAULT 0.0"))
            
        if 'action_logs' in inspector.get_table_names():
            columns = [col['name'] for col in inspector.get_columns('action_logs')]
            if 'ledger_id' not in columns: conn.execute(text("ALTER TABLE action_logs ADD COLUMN ledger_id INTEGER"))

upgrade_database(engine)
SessionLocal = sessionmaker(bind=engine)

# ==========================================
# 2. 核心算法與結算引擎 (Settlement & Odds)
# ==========================================
def calculate_linked_odds(input_odds, margin=1.085):
    """根據輸入主/大賠率和抽水，自動反推客/小賠率"""
    if input_odds <= 1.0: return 0.0
    prob_1 = 1 / input_odds
    prob_2 = margin - prob_1
    if prob_2 <= 0: return 0.0
    return round(1 / prob_2, 3)

def calculate_actual_margin(odds_1, odds_2):
    """計算當前實際抽水值"""
    if odds_1 <= 1 or odds_2 <= 1: return 0.0
    return (1/odds_1) + (1/odds_2)

def settle_asian_handicap(market_type, line, selection, home_score, away_score, home_corners=0, away_corners=0, odds=1.85):
    """亞洲盤口精算引擎 (返回: 狀態, 單位盈虧)"""
    if '大小' in market_type or ('角' in market_type and '讓' not in market_type):
        actual_total = (home_corners + away_corners) if '角' in market_type else (home_score + away_score)
        diff = actual_total - line
        is_over = (selection == '大')
        net_diff = diff if is_over else -diff
    else:
        diff = (home_score - away_score) + line
        is_home = (selection == '主')
        net_diff = diff if is_home else -diff

    if net_diff > 0.25: return 'Win', round(odds - 1.0, 3)
    elif net_diff == 0.25: return 'Half Win', round((odds - 1.0) / 2, 3)
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
    data = [{'odds': t.odds, 'line': t.line, 'target': 1 if t.status in ['Win', 'Half Win'] else 0} for t in tips]
    df = pd.DataFrame(data)
    return df[['odds', 'line']], df['target']

def predict_win_prob_and_kelly(tipster_id, current_odds, current_line, user_bankroll, session, kelly_fraction=0.25):
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
    
    return round(suggested_stake, 2), round(p_win, 4), ai_status

# ==========================================
# 4. UI 視覺格式化函數
# ==========================================
def style_financials(val):
    """將負數標紅，正數標綠"""
    if isinstance(val, (int, float)):
        color = 'red' if val < 0 else 'green' if val > 0 else 'gray'
        return f'color: {color}; font-weight: bold;'
    return ''

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

    tabs = st.tabs([t.name for t in tipsters] + ["💰 全局資金流水"])
    
    for i, tipster in enumerate(tipsters):
        with tabs[i]:
            tips = session.query(Tip).filter(Tip.tipster_id == tipster.id, Tip.is_deleted == False).order_by(Tip.id.desc()).all()
            if not tips:
                st.info(f"{tipster.name} 尚無投注紀錄")
                continue

            data = []
            for t in tips:
                res = f"{t.home_score}-{t.away_score}" if t.status != 'Open' else "未結算"
                if t.status != 'Open' and (t.home_corners or t.away_corners):
                    res += f" (角 {t.home_corners or 0}-{t.away_corners or 0})"
                    
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
            st.dataframe(df.style.map(style_financials, subset=['單位盈虧', '盈虧', '派彩']), use_container_width=True)
            st.markdown(f"**匯總**：總單數 **{len(df)}** | 總投注額 **${df['投注額'].sum():.2f}** | 總盈虧 **${df['盈虧'].sum():.2f}** | 總單位盈虧 **{df['單位盈虧'].sum():.2f}**")
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label=f"📥 匯出 {tipster.name} 報表", data=csv, file_name=f'{tipster.name}_history.csv', mime='text/csv', key=f"dl_{tipster.id}")

    with tabs[-1]:
        ledgers = session.query(BankrollLedger).order_by(BankrollLedger.id.desc()).all()
        if ledgers:
            ldf = pd.DataFrame([{
                '時間': l.created_at.strftime('%Y-%m-%d %H:%M'),
                '帳戶類型': l.account_type,
                '分享者 ID': l.tipster_id if l.tipster_id else 'N/A',
                '金額': l.amount,
                '備註': l.description
            } for l in ledgers])
            st.dataframe(ldf.style.map(style_financials, subset=['金額']), use_container_width=True)
        else:
            st.info("尚無資金流水")

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
                    target_tip.home_score = old_data.get('home_score', None)
                    target_tip.away_score = old_data.get('away_score', None)
                    target_tip.home_corners = old_data.get('home_corners', None)
                    target_tip.away_corners = old_data.get('away_corners', None)
                    
                    if log.ledger_id:
                        target_ledger = session.query(BankrollLedger).get(log.ledger_id)
                        if target_ledger: session.delete(target_ledger)
                        
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
                
                if market == "入球大小": default_line = 2.5
                elif market == "半場入球大小": default_line = 1.5
                elif market == "角球大小": default_line = 9.5
                elif market == "半場角球大小": default_line = 4.5
                else: default_line = 0.0
                
                line = st.number_input("盤口線", value=default_line, step=0.25, key=f"line_{tipster.id}")
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
                user_ledger = db_session.query(BankrollLedger).filter_by(account_type='User').all()
                user_bankroll = sum([l.amount for l in user_ledger]) if user_ledger else 10000.0
                
                rec_stake, win_prob, ai_status = predict_win_prob_and_kelly(
                    tipster.id, st.session_state.odds1, line, user_bankroll, db_session
                )
                
                st.caption(f"模型狀態: {ai_status}")
                st.metric("預測勝率 (RF Model)", f"{win_prob * 100:.1f}%")
                st.metric("建議下注額 (Kelly)", f"${rec_stake}")
                
            with colL:
                stake = st.number_input("實際下注額 (預設為AI建議)", value=float(rec_stake), step=100.0, key=f"stake_{tipster.id}")
                
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
    # [Tab] 資金管理與賽果結算 (t_settle)
    # ----------------------------------------
    with tabs[-2]:
        st.header("資金管理與賽果結算")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.subheader("📝 存取資金操作")
            acc_type = st.radio("對象", ["用家真實資金", "分享者虛擬本金"], horizontal=True)
            target_tipster = None
            if acc_type == "分享者虛擬本金":
                target_tipster = st.selectbox("選擇分享者", tipster_names)
                
            action = st.radio("操作", ["存入 (減少現有資金餘額)", "提取 (回收至現有資金)"], horizontal=True)
            amt_input = st.number_input("金額", min_value=0.0, step=100.0)
            desc = st.text_input("備註 (選填)")
            
            if st.button("提交資金變動"):
                if amt_input > 0:
                    actual_amt = -amt_input if "存入" in action else amt_input
                    db_acc = 'Tipster' if acc_type == "分享者虛擬本金" else 'User'
                    t_id = tipster_dict.get(target_tipster) if target_tipster else None
                    
                    db_session.add(BankrollLedger(account_type=db_acc, tipster_id=t_id, amount=actual_amt, description=desc))
                    db_session.commit()
                    st.success("資金流水已成功紀錄！")
                    st.rerun()

        with col_f2:
            st.subheader("📊 資金總覽")
            user_ledger = db_session.query(BankrollLedger).filter_by(account_type='User').all()
            st.metric("用家真實總資金", f"${sum([l.amount for l in user_ledger]):.2f}")
            
            st.write("各分享者虛擬資金:")
            for t_name, t_id in tipster_dict.items():
                t_ledger = db_session.query(BankrollLedger).filter_by(account_type='Tipster', tipster_id=t_id).all()
                st.caption(f"{t_name}: ${sum([l.amount for l in t_ledger]):.2f}")

        st.divider()
        st.subheader("🏁 待結算注單")
        open_tips = db_session.query(Tip).filter_by(status='Open', is_deleted=False).all()
        
        if not open_tips:
            st.info("目前無待結算注單。")
            
        for t in open_tips:
            with st.expander(f"ID:{t.id} | {t.tournament} | {t.home_team} vs {t.away_team} | {t.market_type} ({t.line}) | 投注額: ${t.stake}"):
                c1, c2, c3, c4, c5 = st.columns(5)
                h_sc = c1.number_input("主隊進球", min_value=0, step=1, key=f"h_sc_{t.id}")
                a_sc = c2.number_input("客隊進球", min_value=0, step=1, key=f"a_sc_{t.id}")
                
                h_cor, a_cor = 0, 0
                if '角' in t.market_type:
                    h_cor = c3.number_input("主隊角球", min_value=0, step=1, key=f"hc_{t.id}")
                    a_cor = c4.number_input("客隊角球", min_value=0, step=1, key=f"ac_{t.id}")
                
                if c5.button("執行結算", key=f"set_{t.id}", type="primary"):
                    status, unit_profit = settle_asian_handicap(t.market_type, t.line, t.selection, h_sc, a_sc, h_cor, a_cor, t.odds)
                    
                    profit = t.stake * unit_profit
                    payout = t.stake + profit if unit_profit >= -0.5 else 0.0 
                    
                    new_ledger = BankrollLedger(
                        account_type='Tipster', tipster_id=t.tipster_id, 
                        amount=profit,
                        description=f"注單 {t.id} 結算 ({status})"
                    )
                    db_session.add(new_ledger)
                    db_session.flush() 
                    
                    old_data = {
                        'status': t.status, 'is_deleted': t.is_deleted, 
                        'unit_profit': t.unit_profit, 'profit': t.profit, 'payout': t.payout,
                        'home_score': t.home_score, 'away_score': t.away_score,
                        'home_corners': t.home_corners, 'away_corners': t.away_corners
                    }
                    db_session.add(ActionLog(
                        action_type='SETTLE', record_id=t.id, ledger_id=new_ledger.id,
                        table_name='tips', old_data=json.dumps(old_data)
                    ))

                    t.home_score, t.away_score = h_sc, a_sc
                    t.home_corners, t.away_corners = h_cor, a_cor
                    t.status = status
                    t.unit_profit = unit_profit
                    t.profit = profit
                    t.payout = payout
                    
                    db_session.commit()
                    st.success(f"結算完成：{status} | 單位盈虧：{unit_profit} | 派彩：${payout}")
                    st.rerun()

    db_session.close()

if __name__ == "__main__":
    main()
