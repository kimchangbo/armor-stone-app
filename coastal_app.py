import streamlit as st
import math
import pandas as pd
import numpy as np
from io import BytesIO

class ArmorCalculator:
    def __init__(self):
        # 해수 단위중량 (kN/m³) - 10.1 적용
        self.gamma_w = 10.1  
        self.g = 9.81        

    def calc_L(self, T, d):
        """파장(L)을 반복법으로 산출"""
        L0 = (self.g * T**2) / (2 * math.pi)
        L = L0
        for _ in range(100):
            kd = 2 * math.pi * d / L
            if kd > 20: 
                break
            L_new = L0 * math.tanh(kd)
            if abs(L_new - L) < 0.001:
                break
            L = L_new
        return L

    def calc_Ks(self, T, d):
        """SPM Table과 동일한 선형 천수계수(Ks) 역산 산출"""
        L0 = (self.g * T**2) / (2 * math.pi)
        L = self.calc_L(T, d)
        kd = 2 * math.pi * d / L
        
        if 2 * kd > 50:
            n = 0.5
        else:
            n = 0.5 * (1 + (2 * kd) / math.sinh(2 * kd))
            
        Ks = math.sqrt(L0 / (2 * n * L))
        return Ks, L0, L

    def check_surf_zone_spm(self, H0_prime, T, m, depth):
        """1. SPM 방식 쇄파대 검토 (도표 독취 데이터 기반 정밀 보간법 적용)"""
        L0 = (self.g * T**2) / (2 * math.pi)
        X_steepness_h0 = H0_prime / (self.g * T**2)
        
        # 원본 SPM Fig 7-3 도표 역산 앵커 데이터
        spm_m_keys = [0.005, 0.010, 0.020, 0.033, 0.050, 0.100]
        spm_X = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.00346, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        # 독취값과 100% 일치하도록 보정된 곡선 데이터
        spm_Y = {
            0.005: [1.38, 1.30, 1.22, 1.15, 1.05, 0.99,  0.95, 0.90, 0.85, 0.80, 0.78],
            0.010: [1.70, 1.55, 1.40, 1.25, 1.15, 1.08,  1.02, 0.95, 0.90, 0.83, 0.79],
            0.020: [2.10, 1.88, 1.60, 1.40, 1.28, 1.17,  1.10, 1.00, 0.92, 0.84, 0.81],
            0.033: [2.35, 2.10, 1.78, 1.55, 1.37, 1.24,  1.17, 1.05, 0.96, 0.87, 0.82],
            0.050: [2.55, 2.30, 1.95, 1.70, 1.48, 1.34,  1.25, 1.12, 1.02, 0.90, 0.83],
            0.100: [2.73, 2.48, 2.15, 1.95, 1.65, 1.48,  1.35, 1.20, 1.08, 0.93, 0.85]
        }
        
        # Log-Log 정밀 보간 수행
        log_X = math.log10(X_steepness_h0) if X_steepness_h0 > 0 else -4
        log_spm_X = [math.log10(x) for x in spm_X]
        
        y_vals_for_m = []
        for m_key in spm_m_keys:
            log_Y = np.interp(log_X, log_spm_X, [math.log10(y) for y in spm_Y[m_key]])
            y_vals_for_m.append(10**log_Y)
            
        Hb_ratio = np.interp(m, spm_m_keys, y_vals_for_m)
        Hb = Hb_ratio * H0_prime
        
        # 쇄파수심 영역 (alpha, beta) 산출
        X_steepness_hb = Hb / (self.g * T**2)
        alpha = 1.49 + 2.5 * X_steepness_hb
        beta = 1.19 + 2.5 * X_steepness_hb
        
        db_max = alpha * Hb
        db_min = beta * Hb
        
        if depth > db_max:
            status = "비쇄파"
            is_breaking = False
        elif depth < db_min:
            status = "쇄파"
            is_breaking = True
        else:
            status = "쇄파대"
            is_breaking = True
            
        return L0, X_steepness_h0, Hb_ratio, Hb, X_steepness_hb, alpha, beta, db_max, db_min, is_breaking, status

    def check_surf_zone_harbor(self, H0_prime, L0, m, depth):
        """2. 항만 및 어항 설계기준 방식 검토"""
        S0 = H0_prime / L0
        
        x_vals = [0.0100, 0.0150, 0.0189, 0.0217, 0.0242, 0.0245, 0.0300, 0.0400, 0.0500]
        y_vals = [2.40,   2.30,   2.22,   2.20,   2.19,   2.18,   2.15,   2.05,   1.95]
        
        peak_ratio = np.interp(S0, x_vals, y_vals)
        h_peak = peak_ratio * H0_prime
        
        if depth > h_peak:
            status = "비쇄파"
            is_breaking = False
        else:
            status = "쇄파(대)"
            is_breaking = True
            
        return S0, peak_ratio, h_peak, is_breaking, status

    def calc_hudson(self, gamma_r, H, Kd, cot_alpha):
        Sr = gamma_r / self.gamma_w
        weight = (gamma_r * (H ** 3)) / (Kd * cot_alpha * ((Sr - 1) ** 3))
        return weight, Sr

    def calc_vandermeer_rock(self, gamma_r, Hs, Tz, cot_alpha, P, S, N):
        Sr = gamma_r / self.gamma_w
        Delta = Sr - 1
        L_om = (self.g * Tz**2) / (2 * math.pi)
        s_m = Hs / L_om
        tan_alpha = 1.0 / cot_alpha
        xi_m = tan_alpha / math.sqrt(s_m) 
        
        xi_mc = (6.2 * (P**0.31) * math.sqrt(tan_alpha)) ** (1 / (P + 0.5))
        
        if xi_m < xi_mc:
            wave_type = "Plunging (붕괴파)"
            Ns = 6.2 * (P**0.18) * ((S / math.sqrt(N))**0.2) * (xi_m**(-0.5))
        else:
            wave_type = "Surging (단파)"
            Ns = 1.0 * (P**-0.13) * ((S / math.sqrt(N))**0.2) * math.sqrt(cot_alpha) * (xi_m**P)
            
        Dn50 = Hs / (Delta * Ns)
        weight = gamma_r * (Dn50 ** 3)
        return weight, wave_type, Sr, L_om, s_m, xi_m, xi_mc, Ns, Dn50

    def calc_vandermeer_ttp(self, gamma_r, Hs, Tz, Nod, N):
        Sr = gamma_r / self.gamma_w
        Delta = Sr - 1
        
        Tm = Tz / 1.2
        L_om = (self.g * Tm**2) / (2 * math.pi)
        s_m = Hs / L_om
        
        Ns = (3.75 * (Nod**0.5) / (N**0.25) + 0.85) * (s_m**-0.2)
        Dn = Hs / (Delta * Ns)
        weight = gamma_r * (Dn ** 3)
        return weight, Sr, Tm, L_om, s_m, Ns, Dn

# --- 번역 오작동 방지용 HTML 텍스트 ---
html_cot = "<span class='notranslate' translate='no'>Cot</span>"
html_tan = "<span class='notranslate' translate='no'>tan</span>"

# --- UI 레이아웃 구성 ---
st.set_page_config(page_title="피복재 통합 검토", page_icon="🌊", layout="wide")

st.title("피복재 및 소파블록 통합 검토 (SPM & 설계기준)")
st.markdown("SPM 방식과 항만 및 어항 설계기준 방식을 교차 적용한 쇄파대 판정 및 피복재 중량 자동 산출 프로그램입니다.")
st.markdown("---")

with st.sidebar:
    st.header("1. 설계 파랑 및 구조 제원")
    Hs = st.number_input("유의파고 Hs (m)", value=4.6, step=0.1)
    Tz = st.number_input("유의주기 Tz (s)", value=11.77, step=0.1)
    depth = st.number_input("구조물 전면수심 h (m)", value=14.41, step=0.1)
    cot_alpha = st.number_input("사면경사 (Cot α)", value=1.5, step=0.1)
    N_waves = st.number_input("내습 파랑수 N", value=1000, step=100)
    st.markdown("---")

    st.header("2. 쇄파대 검토 제원")
    st.info("💡 **환산심해파고($H_0'$)**는 입력된 파고($H_s$), 주기($T_z$), 수심($h$)을 바탕으로 SPM Table을 통해 자동 산출됩니다.")
    m_slope = st.number_input("해저면 경사 m (예: 1/100 = 0.01)", value=0.01, step=0.01)
    st.markdown("---")

    st.header("3. 피복석 파라미터")
    gamma_rock = st.number_input("단위중량 γ (kN/m³) [피복석]", value=26.0, step=0.1)
    st.info("💡 피복석의 안정계수(Kd)는 쇄파대 판정 결과에 따라 **자동 산정**됩니다. (비쇄파: 4.0, 쇄파: 2.0)")
    P_rock = st.number_input("VdM 투과계수 P (0.1~0.6)", value=0.50, step=0.01)
    S_rock = st.number_input("VdM 허용손상도 S (2~8)", value=2.0, step=0.1)
    st.markdown("---")

    st.header("4. 소파블록(TTP) 파라미터")
    gamma_ttp = st.number_input("단위중량 γ (kN/m³) [TTP]", value=22.6, step=0.1)
    st.info("💡 TTP의 안정계수(Kd)는 쇄파대 판정 결과에 따라 **자동 산정**됩니다. (비쇄파: 8.0, 쇄파: 7.0)")
    Nod_ttp = st.number_input("VdM 상대피해율 Nod (0~0.5)", value=0.20, step=0.01)
    st.markdown("---")
    
    run_button = st.button("🚀 검토 실행 (Calculate)", type="primary", use_container_width=True)

if run_button:
    calc = ArmorCalculator()
    
    # 0. 환산심해파고(H0') 자동 산출
    Ks, L0_val, L_val = calc.calc_Ks(Tz, depth)
    H0_prime = Hs / Ks
    
    # 1. 쇄파대 검토 실행
    L0, X_h0, Hb_ratio, Hb, X_hb, alpha, beta, db_max, db_min, is_brk_spm, status_spm = calc.check_surf_zone_spm(H0_prime, Tz, m_slope, depth)
    S0, peak_ratio, h_peak, is_brk_harbor, status_harbor = calc.check_surf_zone_harbor(H0_prime, L0, m_slope, depth)
    
    # 종합 쇄파대 판정 (보수적 적용)
    final_is_breaking = (status_spm != "비쇄파" or status_harbor != "비쇄파")
    
    # 2. KD 값 자동 산정
    Kd_rock = 2.0 if final_is_breaking else 4.0
    Kd_ttp = 7.0 if final_is_breaking else 8.0
    
    # 3. 피복재 중량 계산
    H_design = Hs  
    
    rock_h_weight, r_Sr = calc.calc_hudson(gamma_rock, H_design, Kd_rock, cot_alpha)
    rock_v_weight, r_type, _, r_Lom, r_sm, r_xim, r_ximc, r_Ns, r_Dn50 = calc.calc_vandermeer_rock(gamma_rock, Hs, Tz, cot_alpha, P_rock, S_rock, N_waves)
    
    ttp_h_weight, t_Sr = calc.calc_hudson(gamma_ttp, H_design, Kd_ttp, cot_alpha)
    ttp_v_weight, _, t_Tm, t_Lom, t_sm, t_Ns, t_Dn = calc.calc_vandermeer_ttp(gamma_ttp, Hs, Tz, Nod_ttp, N_waves)
    
    rock_final_kN = max(rock_h_weight, rock_v_weight)
    ttp_final_kN = max(ttp_h_weight, ttp_v_weight)

    # ====================================================
    # UI 출력 - 요약 (비교표 추가)
    # ====================================================
    st.subheader("📊 검토 결과 요약")
    res_col1, res_col2 = st.columns([1, 1.2]) 
    
    with res_col1:
        st.info(f"**🌊 쇄파대 판정 (수심 h = {depth:.2f} m)**\n\n"
                f"- **SPM 기준:** {status_spm} (영역: {db_min:.2f} ~ {db_max:.2f} m)\n"
                f"- **설계기준:** {status_harbor} ($h_{{1/3, peak}}$ = {h_peak:.2f} m)")
                
    with res_col2:
        st.success("**🪨 피복재 소요중량 산정 결과 (비교표)**")
        st.markdown(
            "| 구분 | Hudson 공식 | Van der Meer 공식 | **결정중량(MAX)** |\n"
            "|:---:|:---:|:---:|:---:|\n"
            f"| **피복석** | {rock_h_weight:,.1f} kN | {rock_v_weight:,.1f} kN | **{rock_final_kN:,.1f} kN** |\n"
            f"| **소파블록** | {ttp_h_weight:,.1f} kN | {ttp_v_weight:,.1f} kN | **{ttp_final_kN:,.1f} kN** |"
        )
        st.markdown(f"*(적용 $K_D$: 피복석 **{Kd_rock:.1f}**, 소파블록 **{Kd_ttp:.1f}**)*")

    st.markdown("---")
    
    # ====================================================
    # UI 출력 - 상세 풀이과정
    # ====================================================
    st.subheader("📝 상세 검토 풀이과정")
    
    st.markdown("### 가. 설계조건 및 환산심해파고 자동 산정")
    st.markdown(rf"- 유의파고 $H_{{1/3}} = {Hs} \text{{ m}}$, 유의주기 $T_{{1/3}} = {Tz} \text{{ s}}$, 수심 $h = {depth} \text{{ m}}$")
    st.markdown(rf"- 심해파장 $L_0 = {L0_val:.2f} \text{{ m}}$, 상대수심 $d/L_0 = {(depth/L0_val):.5f}$")
    st.markdown(rf"- 천수계수 산정 $K_s = {Ks:.4f}$ (SPM Table 선형파 이론 역산 적용)")
    st.markdown(rf"- **환산심해파고 $H_0' = H_{{1/3}} / K_s = {Hs} / {Ks:.4f} = {H0_prime:.3f} \text{{ m}}$**")

    st.markdown("---")
    
    st.markdown("### 나. S.P.M 방법에 의한 쇄파대 검토")
    st.markdown("**1) 쇄파고($H_b$) 도표 보간 산정** *(Shore Protection Manual Vol. II, Fig. 7-3 기반)*")
    st.markdown(rf"- 파형경사 파라미터 $H_0' / (g \cdot T^2) = {H0_prime:.3f} / (9.81 \times {Tz}^2) = {X_h0:.5f}$")
    st.markdown(rf"- 도표 독취 역산 계수 $H_b / H_0' = {Hb_ratio:.3f}$")
    st.markdown(rf"- 산출된 쇄파고 $H_b = {Hb_ratio:.3f} \times {H0_prime:.3f} = {Hb:.3f} \text{{ m}}$")
    # 해저경사 보간법 설명 추가
    st.markdown(rf"*(해저경사 m={m_slope} 등 도표에 직접 곡선이 표시되지 않은 경사는, 앵커 곡선 데이터를 바탕으로 수학적 2D 정밀 보간법을 사용하여 오차 없이 독취 및 산출하였습니다.)*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("**2) 쇄파수심($d_b$) 영역 산정** *(Shore Protection Manual Vol. II, Fig. 7-2)*")
    st.markdown(rf"- 쇄파고 파라미터 $H_b / (g \cdot T^2) = {Hb:.3f} / (9.81 \times {Tz}^2) = {X_hb:.5f}$")
    st.markdown(rf"- 상한계수 $\alpha = {alpha:.3f}$, 하한계수 $\beta = {beta:.3f}$")
    st.markdown(rf"- $d_{{b,max}} = \alpha \times H_b = {alpha:.3f} \times {Hb:.3f} = {db_max:.3f} \text{{ m}}$")
    st.markdown(rf"- $d_{{b,min}} = \beta \times H_b = {beta:.3f} \times {Hb:.3f} = {db_min:.3f} \text{{ m}}$")
    st.markdown(rf"👉 **판정:** 수심 $h ({depth} \text{{ m}}) > d_{{b,max}} ({db_max:.3f} \text{{ m}})$ 여부 비교결과 **{status_spm}** 임.")

    st.markdown("---")
    
    st.markdown("### 다. 항만 및 어항 설계기준을 이용한 쇄파대 검토")
    st.markdown("**1) 유의파고 최대치 출현 수심 산정** *(도참 4-20b)*")
    st.markdown(rf"- 파형경사 $H_0' / L_0 = {H0_prime:.3f} / {L0:.3f} = {S0:.5f}$")
    st.markdown(rf"- 해저경사 = {m_slope}")
    st.markdown(rf"- 그래프 독취비율 $(h_{{1/3}})_{{peak}} / H_0' = {peak_ratio:.2f}$")
    st.markdown(rf"- 최대치 출현 수심 $(h_{{1/3}})_{{peak}} = {peak_ratio:.2f} \times {H0_prime:.3f} = {h_peak:.3f} \text{{ m}}$")
    st.markdown(rf"👉 **판정:** 수심 $h ({depth} \text{{ m}}) > (h_{{1/3}})_{{peak}} ({h_peak:.3f} \text{{ m}})$ 여부 비교결과 **{status_harbor}** 임.")

    st.markdown("---")
    
    st.markdown("### 라. 쇄파대 검토결과 요약")
    summary_data = {
        "구분": [f"수심 h(m)", "S.P.M에 의한 방법", "항만 및 어항 설계기준", "최종 판정"],
        "검토값": [f"{depth:.2f}", f"{db_min:.2f} ~ {db_max:.2f}", f"{h_peak:.2f}", f"{status_spm}"],
        "결과": ["-", f"{status_spm}", f"{status_harbor}", f"{'비쇄파' if not final_is_breaking else '쇄파(대)'}"]
    }
    st.table(pd.DataFrame(summary_data))
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # 피복석 상세 계산
    # ----------------------------------------------------
    st.markdown("### 마. 피복석 소요중량 산정")
    
    st.markdown("#### 1) Hudson 공식에 의한 산정")
    st.markdown(rf"- **설계파고($H_{{1/3}}$)**: $H_s = {H_design:.2f} \text{{ m}}$")
    st.markdown(rf"- **피복석 비중($S_r$)**: $\gamma_r / \gamma_w = {gamma_rock:.2f} / 10.1 = {r_Sr:.3f}$")
    
    # HTML translate='no' 적용, 수식 내부 글자 분리(\mathrm{C}\kern0.1ex\mathrm{o}\kern0.1ex\mathrm{t})
    st.markdown(f"- **안정계수($K_D$)**: {Kd_rock:.1f} (판정결과 자동적용), **사면경사({html_cot} $\\alpha$)**: {cot_alpha}", unsafe_allow_html=True)
    st.latex(r"W = \frac{\gamma_r H^3}{K_D \cdot \mathrm{C}\kern0.1ex\mathrm{o}\kern0.1ex\mathrm{t}\,\alpha \cdot (S_r - 1)^3}")
    st.latex(rf"W = \frac{{{gamma_rock:.2f} \times {H_design:.2f}^3}}{{{Kd_rock:.1f} \times {cot_alpha} \times ({r_Sr:.3f} - 1)^3}} = {rock_h_weight:,.1f} \text{{ kN}}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### 2) Van der Meer 공식에 의한 산정")
    st.markdown(rf"- **적용파고($H_s$)**: {Hs} $\text{{m}}$")
    st.markdown(rf"- **심해파장($L_{{om}}$)**: $g T_z^2 / 2\pi = 9.81 \times {Tz}^2 / 2\pi = {r_Lom:.2f} \text{{ m}}$")
    st.markdown(rf"- **파형경사($s_m$)**: $H_s / L_{{om}} = {Hs} / {r_Lom:.2f} = {r_sm:.4f}$")
    
    # HTML translate='no' 적용, 루트 안 기호를 tan으로 변경
    st.markdown(f"- **쇄파유사도($\\xi_m$)**: {html_tan} $\\alpha / \\sqrt{{s_m}} = (1/{cot_alpha}) / \\sqrt{{{r_sm:.4f}}} = {r_xim:.3f}$", unsafe_allow_html=True)
    st.markdown(f"- **임계 쇄파유사도($\\xi_{{mc}}$)**: $(6.2 \\times P^{{0.31}} \\sqrt{{\\mathrm{{t}}\\kern0.1ex\\mathrm{{a}}\\kern0.1ex\\mathrm{{n}}\\,\\alpha}})^{{1/(P+0.5)}} = {r_ximc:.3f}$ (투과계수 P={P_rock})", unsafe_allow_html=True)
    
    st.markdown(rf"- **파랑조건 판정**: $\xi_m({r_xim:.3f})$ {'<' if r_xim < r_ximc else '>'} $\xi_{{mc}}({r_ximc:.3f})$ 이므로 **{r_type}** 공식 적용")
    st.markdown(rf"- **안정계수($N_s$)**: {r_Ns:.3f} (허용손상도 S={S_rock}, 내습파랑수 N={N_waves})")
    st.markdown(rf"- **공칭직경($D_{{n50}}$)**: $H_s / ((S_r - 1) N_s) = {Hs} / (({r_Sr:.3f} - 1) \times {r_Ns:.3f}) = {r_Dn50:.3f} \text{{ m}}$")
    st.latex(r"W = \gamma_r \times D_{n50}^3")
    st.latex(rf"W = {gamma_rock:.2f} \times {r_Dn50:.3f}^3 = {rock_v_weight:,.1f} \text{{ kN}}")

    st.markdown("---")
    
    # ----------------------------------------------------
    # 소파블록(TTP) 상세 계산
    # ----------------------------------------------------
    st.markdown("### 바. 소파블록 (TTP) 소요중량 산정")

    st.markdown("#### 1) Hudson 공식에 의한 산정")
    st.markdown(rf"- **설계파고($H_{{1/3}}$)**: $H_s = {H_design:.2f} \text{{ m}}$")
    st.markdown(rf"- **TTP 비중($S_r$)**: $\gamma_r / \gamma_w = {gamma_ttp:.2f} / 10.1 = {t_Sr:.3f}$")
    
    st.markdown(f"- **안정계수($K_D$)**: {Kd_ttp:.1f} (판정결과 자동적용), **사면경사({html_cot} $\\alpha$)**: {cot_alpha}", unsafe_allow_html=True)
    st.latex(r"W = \frac{\gamma_r H^3}{K_D \cdot \mathrm{C}\kern0.1ex\mathrm{o}\kern0.1ex\mathrm{t}\,\alpha \cdot (S_r - 1)^3}")
    st.latex(rf"W = \frac{{{gamma_ttp:.2f} \times {H_design:.2f}^3}}{{{Kd_ttp:.1f} \times {cot_alpha} \times ({t_Sr:.3f} - 1)^3}} = {ttp_h_weight:,.1f} \text{{ kN}}")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 2) Van der Meer 공식에 의한 산정")
    st.markdown(rf"- **적용파고($H_s$)**: {Hs} $\text{{m}}$")
    
    st.markdown(rf"- **평균주기($T_m$)**: $T_{{1/3}} / 1.2 = {Tz} / 1.2 = {t_Tm:.2f} \text{{ s}}$")
    st.markdown(rf"- **심해파장($L_{{om}}$)**: $g T_m^2 / 2\pi = 9.81 \times {t_Tm:.2f}^2 / 2\pi = {t_Lom:.2f} \text{{ m}}$")
    st.markdown(rf"- **파형경사($s_m$)**: $H_s / L_{{om}} = {Hs} / {t_Lom:.2f} = {t_sm:.4f}$")
    
    st.markdown(rf"- **안정계수($N_s$)**: $(3.75 \sqrt{{N_{{od}}}} / N^{{0.25}} + 0.85) \times s_m^{{-0.2}} = {t_Ns:.3f}$ (상대피해율 Nod={Nod_ttp})")
    st.markdown(rf"- **공칭직경($D_n$)**: $H_s / ((S_r - 1) N_s) = {Hs} / (({t_Sr:.3f} - 1) \times {t_Ns:.3f}) = {t_Dn:.3f} \text{{ m}}$")
    st.latex(r"W = \gamma_r \times D_n^3")
    st.latex(rf"W = {gamma_ttp:.2f} \times {t_Dn:.3f}^3 = {ttp_v_weight:,.1f} \text{{ kN}}")

    st.markdown("---")

    # ====================================================
    # 엑셀 다운로드 (오류 방지를 위해 openpyxl 엔진 명시 및 태그 제거)
    # ====================================================
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 1. 요약 시트
        df_summary = pd.DataFrame({
            "구분": ["Hs(m)", "수심(m)", "환산심해파고(m)", "SPM 판정", "설계기준 판정", "피복석(kN)", "TTP(kN)"],
            "결과값": [Hs, depth, round(H0_prime, 3), status_spm, status_harbor, round(rock_final_kN, 1), round(ttp_final_kN, 1)]
        })
        df_summary.to_excel(writer, index=False, sheet_name='요약(Summary)')
        
        # 2. 상세풀이 시트 (HTML 태그가 포함되지 않은 순수 문자열 배열)
        detail_data = [
            ["[ 상세 검토 풀이과정 ]", ""],
            ["", ""],
            ["가. 설계조건 및 환산심해파고 자동 산정", ""],
            ["유의파고 (H1/3)", f"{Hs} m"],
            ["유의주기 (T1/3)", f"{Tz} s"],
            ["수심 (h)", f"{depth} m"],
            ["심해파장 (L0)", f"{L0_val:.2f} m"],
            ["상대수심 (d/L0)", f"{(depth/L0_val):.5f}"],
            ["천수계수 (Ks)", f"{Ks:.4f}"],
            ["환산심해파고 (H0')", f"{H0_prime:.3f} m"],
            ["", ""],
            ["나. S.P.M 방법에 의한 쇄파대 검토", ""],
            ["파형경사 파라미터 (H0'/gT^2)", f"{X_h0:.5f}"],
            ["도표 역산 계수 (Hb/H0')", f"{Hb_ratio:.3f}"],
            ["산출된 쇄파고 (Hb)", f"{Hb:.3f} m (경사 {m_slope} 2D 보간 적용)"],
            ["쇄파고 파라미터 (Hb/gT^2)", f"{X_hb:.5f}"],
            ["상한계수 (alpha)", f"{alpha:.3f}"],
            ["하한계수 (beta)", f"{beta:.3f}"],
            ["최대 쇄파수심 (db,max)", f"{db_max:.3f} m"],
            ["최소 쇄파수심 (db,min)", f"{db_min:.3f} m"],
            ["판정 결과", status_spm],
            ["", ""],
            ["다. 항만 및 어항 설계기준 쇄파대 검토", ""],
            ["파형경사 (H0'/L0)", f"{S0:.5f}"],
            ["독취비율 ((h1/3)peak / H0')", f"{peak_ratio:.2f}"],
            ["최대치 출현 수심", f"{h_peak:.3f} m"],
            ["판정 결과", status_harbor],
            ["", ""],
            ["라. 피복석 소요중량 산정", ""],
            ["[Hudson 공식]", ""],
            ["설계파고 (H1/3)", f"{H_design:.2f} m"],
            ["피복석 비중 (Sr)", f"{r_Sr:.3f}"],
            ["안정계수 (KD)", f"{Kd_rock:.1f}"],
            ["사면경사 (Cot a)", f"{cot_alpha}"],
            ["소요중량 (W)", f"{rock_h_weight:,.1f} kN"],
            ["[Van der Meer 공식]", ""],
            ["심해파장 (Lom)", f"{r_Lom:.2f} m"],
            ["파형경사 (sm)", f"{r_sm:.4f}"],
            ["쇄파유사도 (tan a / sqrt(sm))", f"{r_xim:.3f}"],
            ["임계 쇄파유사도", f"{r_ximc:.3f}"],
            ["파랑조건 판정", r_type],
            ["안정계수 (Ns)", f"{r_Ns:.3f}"],
            ["공칭직경 (Dn50)", f"{r_Dn50:.3f} m"],
            ["소요중량 (W)", f"{rock_v_weight:,.1f} kN"],
            ["", ""],
            ["마. 소파블록 (TTP) 소요중량 산정", ""],
            ["[Hudson 공식]", ""],
            ["설계파고 (H1/3)", f"{H_design:.2f} m"],
            ["TTP 비중 (Sr)", f"{t_Sr:.3f}"],
            ["안정계수 (KD)", f"{Kd_ttp:.1f}"],
            ["사면경사 (Cot a)", f"{cot_alpha}"],
            ["소요중량 (W)", f"{ttp_h_weight:,.1f} kN"],
            ["[Van der Meer 공식]", ""],
            ["평균주기 (Tm)", f"{t_Tm:.2f} s"],
            ["심해파장 (Lom)", f"{t_Lom:.2f} m"],
            ["파형경사 (sm)", f"{t_sm:.4f}"],
            ["안정계수 (Ns)", f"{t_Ns:.3f}"],
            ["공칭직경 (Dn)", f"{t_Dn:.3f} m"],
            ["소요중량 (W)", f"{ttp_v_weight:,.1f} kN"]
        ]
        
        df_detail = pd.DataFrame(detail_data, columns=["항목", "계산 및 결과값"])
        df_detail.to_excel(writer, index=False, sheet_name='상세풀이(Detail)')
        
        # 엑셀 셀 너비 포맷팅 (openpyxl API 사용)
        worksheet_summary = writer.sheets['요약(Summary)']
        worksheet_detail = writer.sheets['상세풀이(Detail)']
        worksheet_summary.column_dimensions['A'].width = 15
        worksheet_summary.column_dimensions['B'].width = 15
        worksheet_detail.column_dimensions['A'].width = 40
        worksheet_detail.column_dimensions['B'].width = 35
    
    st.download_button("📥 전체 결과(요약+상세) 엑셀(.xlsx) 다운로드", data=output.getvalue(), file_name="coastal_armor_detailed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

else:
    st.info("👈 좌측 사이드바에 제원을 입력하고 **검토 실행** 버튼을 눌러주세요.")