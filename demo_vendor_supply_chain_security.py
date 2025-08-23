"""
Vendor and Supply Chain Security System Demo
Demonstrates comprehensive vendor security management capabilities
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from security.vendor_supply_chain.vendor_security_assessor import (
    VendorSecurityAssessor, VendorProfile, RiskLevel
)
from security.vendor_supply_chain.vulnerability_scanner import (
    ThirdPartySoftwareScanner, VulnerabilitySeverity
)
from security.vendor_supply_chain.sbom_manager import (
    SBOMManager, SBOMFormat
)
from security.vendor_supply_chain.vendor_access_monitor import (
    VendorAccessMonitor, AccessType
)
from security.vendor_supply_chain.incident_tracker import (
    VendorIncidentTracker, IncidentSeverity, IncidentCategory
)
from security.vendor_supply_chain.contract_templates import (
    SecurityContractTemplateManager, ContractType, ComplianceFramework
)

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title: str):
    """Print formatted section"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

async def demo_vendor_security_assessment():
    """Demonstrate vendor security assessment capabilities"""
    print_header("VENDOR SECURITY ASSESSMENT DEMO")
    
    assessor = VendorSecurityAssessor()
    
    # Create sample vendor profiles
    vendors = [
        VendorProfile(
            vendor_id="VENDOR-CLOUD-001",
            name="SecureCloud Solutions",
            contact_email="security@securecloud.com",
            business_type="cloud_provider",
            services_provided=["data_processing", "backup", "analytics"],
            data_access_level="confidential",
            compliance_certifications=["SOC2_TYPE_II", "ISO27001", "PCI_DSS"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        ),
        VendorProfile(
            vendor_id="VENDOR-SAAS-002",
            name="DataFlow Analytics",
            contact_email="support@dataflow.com",
            business_type="software_vendor",
            services_provided=["analytics", "reporting"],
            data_access_level="internal",
            compliance_certifications=["SOC2_TYPE_II"],
            created_at=datetime.now(),
            last_updated=datetime.now()
        ),
        VendorProfile(
            vendor_id="VENDOR-RISKY-003",
            name="QuickFix Solutions",
            contact_email="info@quickfix.com",
            business_type="consulting",
            services_provided=["support", "maintenance"],
            data_access_level="restricted",
            compliance_certifications=[],  # No certifications - high risk
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
    ]
    
    print_section("Conducting Vendor Security Assessments")
    
    assessments = []
    for vendor in vendors:
        print(f"\n📋 Assessing vendor: {vendor.name}")
        print(f"   Business Type: {vendor.business_type}")
        print(f"   Data Access Level: {vendor.data_access_level}")
        print(f"   Certifications: {', '.join(vendor.compliance_certifications) if vendor.compliance_certifications else 'None'}")
        
        assessment = await assessor.assess_vendor(vendor)
        assessments.append(assessment)
        
        print(f"   ✅ Assessment completed")
        print(f"   Risk Score: {assessment.risk_score:.2f}/10")
        print(f"   Risk Level: {assessment.risk_level.value.upper()}")
        print(f"   Findings: {len(assessment.findings)} categories analyzed")
        print(f"   Recommendations: {len(assessment.recommendations)} items")
        
        # Show key recommendations
        if assessment.recommendations:
            print(f"   Key Recommendations:")
            for i, rec in enumerate(assessment.recommendations[:3], 1):
                print(f"     {i}. {rec}")
    
    print_section("Assessment Summary Report")
    
    # Generate summary statistics
    total_assessments = len(assessments)
    risk_distribution = {}
    for assessment in assessments:
        risk_level = assessment.risk_level.value
        risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
    
    avg_risk_score = sum(a.risk_score for a in assessments) / total_assessments
    
    print(f"📊 Assessmenin())(mancio.runsy    a:
__main__"__ == "__name

if _exc()eback.printtracck
        ebaort trac    imp    ")
tr(e)}r: {san erroered encount❌ Demo t(f"\nrin:
        pion as ept Except   exce     
 ds!")
   tandarindustry shat exceed s tilitiecapabe-grade enterpris       print(f"
     ates")y demonstrllcessfuuc s SystemSecurity Vendor  ScrollIntelt(f"\n🏆       prin      
 t}")
   {benefiint(f"   •       prs:
     fitit in beneor benef 
        f    
   ]"
        lowsns workfioty operatcuriseting  existhrates wi  "Integ        tc.)",
  PR, HIPAA, e(SOC2, GDiance ompllatory cupports regu"S      t",
      t managemeny requiremenract securitcontnes reamli  "St
          ",itigationand threat mponse nt res incides proactive    "Enable",
        ity postureurecendor sy into vitsibil vil-timees reaid  "Prov          ,
meworks"iple fraainst multation agiance valid complmatesuto         "A80%",
   ent time by ssessmr security avendoeduces        "R = [
     fits bene
       ts:")prise Benefi"\n🎯 Enter   print(     
        
")capability}}. {   {i   print(f":
          1)lities,pabimerate(caty in enu i, capabili   for          
      ]
t"
     nagemenle mador lifecyc ven"End-to-end            ing",
 reportandvalidation ce omplianme c "Real-ti          s",
 ntractcor vendor mplates foement terequiry Securit      "",
      wskflosponse word reant tracking ncidend iomate "Aut      
     tion",omaly detec anwithtoring access monior  vendmitedime-li       "Tt",
     en and managem) generation(SBOMerials l of Mate Biloftwar"S            tion",
 detecith backdoorng wlity scanniulnerabiprehensive vCom        "",
     scoringnhanced risks with AI-esessmentasy curitvendor seAutomated    "         ities = [
capabil")
        nstrated:emoes Dlitiapabi"\n🚀 Key C print(     
     ")
     cler lifecye vendo Completflow: Work• End-to-Endint(f"   pr       
 ")emonstratedypes d tpletis: Mullate Temp • Contractt(f"  in   pr
     naged")ts)} malen(incidens: {y Incidenturit   • Sec   print(f"ed")
     arios testtiple scenrants: Muls G• Acces   "int(fpr  
      reated")(sboms)} cted: {leneraSBOMs Gen"   •     print(fmed")
    s)} perforultan_res(sc Scans: {lenbility Vulnerant(f"   •        primpleted")
sments)} co{len(assesssessments: r A"   • Vendo   print(f)
     atistics:"emo St("\n📊 D print   y!")
    lled successfus demonstratonentmpurity cohain sec clyendor supp"✅ All v  print(  
      
      UMMARY")ION SLET"DEMO COMPader( print_he    
   mmaryFinal su  #     
      flow()
    ion_worktegratdemo_in await lt =suow_re    workfl...")
    nstrationflow demoworkegrated int Running print("\n🎯
        demorkflow rated won integ  # Ru   
      )
     plates(ntract_temcowait demo_er = amanagntract_      co)
  ent(management_emo_incid await dcidents =     inoring()
   onitccess_m_vendor_ait demo= awa_monitor     accesst()
    agemenmo_sbom_manit de awa   sboms =    
 ning()bility_scanera demo_vulnwaitlts = aan_resu        sct()
sessmenty_asndor_securimo_vet de = awaiments      assess
         .")
 rations..ent demonstal componndividuning i"\n🎯 Run    print(   os
  demnentcompoual vid # Run indi      try:
     s")
    
ionratsecurity opeced hanwith AI-enards y standdustreeding in("Exc    printies")
pabilit caanagementy mor securitde venderprise-grag entemonstratin"D print(")
   SYSTEMTY N SECURICHAI& SUPPLY L VENDOR "SCROLLINTE_header( print  
  demo"""itychain securpply  vendor suprehensiveun com   """Rf main():
 ync de
ass
    }
mendationns": recomcommendatio      "re_risk,
  ": overallkall_risver"o    ,
    ": incidentncident "i   est,
    equ": access_rs_request  "accesm,
      sbo:     "sbom"  esult,
  t": scan_r_resul     "scannt,
   : assessmeessment""ass        ile,
of": vendor_pr_profileorend   "v{
     turn  
    re")
   ccessfully!pleted suflow comty workdor securi-end vennd-ton✅ Et(f"\   prin")
    
  {rec}}."     {irint(f      p1):
  ndations, (recommete in enumerarec    for i, ")
mendations: Final Recom   print(f"
   g")
    rinlar monitontinue reguments - corecurity requiets seVendor mens.append("mmendatio     recoons:
   ti recommenda
    if not
    e")omaly rath anhigring due to ccess monitoEnhance append("s.aommendation   rec
     10:ly_rate'] > anomanalysis']['isk_ass_report['r
    if acce   ftware")
  so vendorinrs ndicatokdoor itial bacigate potenstend("Inves.appmmendationreco      0:
   ndicators) >or_ickdoult.bascan_resn( 
    if leent")
   oymepl before dilitiesrabty vulnere securitwaress sofAdd"pend(tions.apecommenda       r 5.0:
 _score >iskerall_rult.ov scan_res
    if
    ")tingrah risk  hig due to assessmenturityitional secConduct adds.append("endationmm     recoCAL]:
   CRITIl.GH, RiskLevevel.HI [RiskLelevel insk_nt.rissme
    if asse    []
 s =mmendation   recotions
 recommendae final at# Gener  
    
  ore:.3f}")tle()}: {sc).tice('_', ' 'tor.repla"     • {fac(f    printems():
    rs.itn risk_facto ir, scorefactofor 
    ors:")k Factf"   Ris
    print(3f}")erall_risk:. {ove: Risk Scorall Vendor   Over  print(f"
    
  isk_factors)s()) / len(rrs.valuetok_facum(ris= sk rall_ris   
    ove
    }
 0) 1.nts'] / 10,total_incidesummary'][''incident_nt_summary[in(incideequency": mident_frinc     "
   ate'] / 100,ly_rnomaalysis']['aisk_anport['r access_reomalies":ess_an"acc        / 10,
ore _scl_riskveralt.oules": scan_ruritysoftware_sec     ",
   _score / 10t.riskmenassessnt": ssmeecurity_asse "s     = {
   rs_facto   riskk
  vendor risverallalculate o   
    # C")
 Assessmenterall Risk  Phase 7: Ovrint(f"\n📈s
    pndationnd RecommeSummary ak : Ris Phase 7   #
    
 ")ts']} incidencidents]['total_inmary'umdent_sinciy['dent_summar: {incimaryt Sum   Incidenf"
    print(vities")} actictivities']']['total_aess_summarycceport['a {access_rss Report: Acce print(f"  ")
   ed analyzponentsnts']} com_compone']['totalsummaryics['m_analyt {sbo Analytics:SBOMf"   
    print(")mmendationss'])} recoionendatrecommt['n(scan_reporport: {leScan Ref"      print(")
 ndingsngs'])} fidit['finnt_reporsme{len(asses: portment Ressess"   Ant(f")
    priessfullyd succenerateeports g   ✅ All rf"int(   pr
    
 0)or_id, 3profile.vend(vendor_dent_summaryor_incinerate_vendtracker.gencident_await iy = _summar   incident_id, 7)
 ndorile.ve(vendor_profrtaccess_repoerate_r.geness_monito= await acct reporccess_m_id)
    am.sbobocs(sti_sbom_analyager.getait sbom_manlytics = awanabom_lt)
    ssun_re(scart_reposcane_generatr. scanneait = aweport
    scan_r(assessment)ortment_repe_assessor.generatssit asse = awat_reportassessments
    l repor Generate al #
    
   g")portin Remprehensive6: Coe "\n📊 Phasprint(f    eporting
hensive Rompre 6: C
    # Phase)
    solved"nt reide(f"   ✅ Inc 
    print
   t"
    )ac impcurity- no seance d maintenveo approt was due triftion dfigurates="Con no,
       "any.comomp@canalystrity_ter="secu     updaSOLVED,
   tStatus.REdenker.Incient_tractus=incid  new_staid,
      t.incident_idenncnt_id=i  incide  s(
    ent_statue_incidker.updatt_traccidenin
    await     )
    stem"
er="sy      assign  om",
ompany.cst@calyy_an"securitignee=        ass,
ent_idcident.incid_id=inincident        ncident(
ssign_iker.aent_tracidt inc  await
   incidennd resolveAssign a  # 
  )
    us.value}"atnt.stideus: {inc   Statprint(f"ue}")
    severity.valncident.everity: {i   S(f"rint")
    pid}incident_: {incident.t ID  Inciden" int(f    preated")
cident crt(f"   ✅ Inprin
       
 )
    on"]tiguraonfim_csystepes=["ta_ty affected_da    
   ment"],ion_manage"configuratstems=[_syfected af       om",
em@company.cng_systritomoniporter="     re.LOW,
   veritydentSeity=Incisever     
   TION,NCE_VIOLAPLIAOMntCategory.Cy=Incide categor       w",
e windoenancintved maapproe id outstion changesed configuraoring detectitmontomated on="Au descripti  ed",
     Detectft  Driiguration="Conftle  time,
      _profile.naame=vendor_n    vendor
    id,ndor_le.veor_profidor_id=vend ven     ident(
  e_increatnt_tracker.ct incide awaiincident =    tration
or demonsincident fa minor eate # Cr    
  t")
  Managemen5: Incident "\n🚨 Phase f    print(ulation
t Sim Incidenase 5:  # Ph 
  
   ") approved"   ✅ Access  print(f   )
      om"
     @company.cy_managerrit"secu   approver=        ,
 request_idss_request.uest_id=acce       req   
  s_request(e_accesor.approv_monitwait accessuccess = al_sva    approd:
    uireapprover_reqrequest. access_    ifequired
cess if rrove ac# App  
  }")
    edirover_requpr_request.aped: {accessquirroval Rent(f"   App   pri}")
 s.valueequest.statuess_rcc{atatus:   Srint(f" 
    p")t_id}equesss_request.r: {acceequest ID"   Rnt(f  priated")
  quest crereAccess "   ✅  print(f   
 
    )
   y=False emergenc     s=6,
  hour   duration_on",
     izatie and optimancem maintenarterly systation="Quiciness_justifus b
       on"],figurati, "conta"alytics_daurces=["an      resoD_WRITE,
  REA=AccessType. access_type
       otech.com",="admin@demester_emailequ        rdor_id,
venfile.rodor_pendor_id=ven  v   
   s(uest_acceseq_monitor.raccessst = await s_reque    acces
    
t")gemencess Manae 4: Ac\n🔐 Phas print(f"gement
   Manaccess  Phase 4: A   #)
    
 sing_ok=Truemisth).unlink(ware_pa  Path(soft     :
     finally    
)
    alue}"mat.v.fort: {sbom"   Forma print(f")
       s)}entsbom.componnents: {len("   Compont(f       pri")
 generated   ✅ SBOM    print(f"      
       )
    
    "1"2.3.are_version=   softw,
         ics"ech Analytame="DemoT  software_n
          path,are_twh=sofe_pat    softwar
        .vendor_id,or_profile=vendor_idvend      m(
      rate_sboager.gene_man sbom = await    sbom   n
 neratio  # SBOM ge             
rs)}")
 _indicatockdoor_result.baen(scan {lndicators:r Idoof"   Back   print()}")
     iesrabilitulne_result.vs: {len(scannerabilitie"   Vulrint(f   p
     ")f}/10:.2l_risk_scoreoveralt. {scan_resulScore:"   Risk nt(f   pri
     pleted") comanty scbili✅ Vulneraint(f"         pr      
    )
       
 2.3.1"version="are_     softw  ,
     cs"timoTech Analyre_name="De   softwa         path,
are_path=softwware_       soft
     e.vendor_id,filroendor_pndor_id=v    ve(
        ackageware_psoftn_ scanner.sca= awaitt   scan_resuln
       scaerabilityVuln     # try:
   
    
    .nameilemp_fth = teftware_pa
        so")    ""  .now()}
  atetimep": dtames, "timsed": True"proces   return {     ics data
lyt Process ana        # data):
_data(self, process
    defom"
    emotech.cpi.d"https://a .base_url =      selfpi_key
  = alf.api_key        se
 :i_key)(self, ap_init__    def _ssor:
alyticsProce

class Andatetime import datetimes
from uestrt reqashlib
impo himports Module
alytic DemoTech An
#e(""".writ temp_file    _file:
   e) as tempFalspy', delete='., suffix=w'e='(modledTemporaryFifile.Name   with tempage
 ackware p softreate sample
    # C)
    is"ty AnalysSecuriware Softn🔍 Phase 3: print(f"\  nalysis
   Security A Software 3:   # Phase    
 } items")
ons)atiendcommessment.reassons: {len(tidaRecommen  (f" rint")
    psriegs)} categoment.findin(assessings: {lenf"   Find
    print(r()}").value.uppelevelnt.risk_{assessmeel: ev  Risk L print(f" 10")
   ore:.2f}/sk_scessment.ricore: {asssk S(f"   Ri
    printleted")ompAssessment c   ✅ print(f"
        ofile)
r(vendor_prs_vendoessor.assesawait assnt =  assessme  
   ")
  tAssessmeny itse 2: Secur"\n🔍 Pha(frint
    pessmentrity Ass Secuse 2:
    # Pha
    ")ementss']} requiruirement_reqotal_language['tctntrawith {coated tract generConint(f"   ✅ 
    pr    
    )
es"cs servicanalyting and ssi proceatan="Cloud diptioscre_de    servic.name,
    rofilendor_pver_name=    vendod,
    ate_implemplate.te_id=t template(
       act_languagerate_contrger.geneontract_manaage = cct_langu  contra]
    
  mplates[0oud_teate = cl
    templERVICE)OUD_SactType.CLype(Contr_by_t_templatesetmanager.gcontract_s = mplatete
    cloud_")
    ceand Complianeneration ract Ghase 1: Contn📋 Pint(f"\
    prnce Checkd Compliation anct GeneratraConPhase 1: 
    # 
    el}")access_levta_r_profile.dandoevel: {veta Access Lint(f"   Dapr  type}")
  ess_e.businfilor_proype: {vendsiness TBu(f"   int)
    prndor_id}"rofile.ve: {vendor_p  Vendor IDt(f"  prinme}")
   ofile.na{vendor_prndor: ing vet(f"🏢 Manag prin    )
    
e.now()
   ated=datetimst_upd   la
     e.now(),=datetimd_at      create
  "],01"ISO270, YPE_II""SOC2_Ttions=[certificacompliance_",
        ntialconfideess_level="  data_acc   ,
   kup"]s", "bacnalyticng", "aessia_proc=["datedprovidervices_     s
   der",oviud_prs_type="clo     busines
   tech.com",urity@demoect_email="s  contac,
      lutions"ch SoDemoTeme="   na  ",
   001VENDOR-DEMO-vendor_id="     ofile(
   e = VendorProfilendor_prion
    vinformat Vendor 
    #    Manager()
latetractTempcurityCon Senager = contract_ma   acker()
orIncidentTr= Vendracker nt_tdencir()
    iessMonitorAcc= Vendomonitor 
    access_r()= SBOMManagenager _ma sbomner()
   reScanartySoftwairdPr = Thanne    scor()
sstyAsseVendorSecuriassessor =    s
 ent componlllize anitia # I   t")
    
nagemen Ma Lifecycledorlete Ven("Comprint_section  
    pLOW")
  TY WORKFOR SECURI VEND"END-TO-ENDer(adrint_he""
    p"owy workfl securitend vendorrate end-to-onst"""Dem 
   ow():orkfl_wionintegrat demo_nc defsy
a
nager contract_maturn    
    re}")
'])mary['sla_summatrix: {len(A Metrics    SLrint(f" 
    ppping'])}")_maceix['complians: {len(matre Mappingnc Complia   rint(f"  p")
   egory'])}_by_catuirements'reqx[ {len(matriories:   Categf"  int(   pr")
 ted:neraGetrix nts Mauireme"   Req print(f
    
   id)te.template__templaudlo_matrix(cequirementerate_rr.genmanagetract_ = conmatrixmatrix
    equirements # Generate r
    
    ")gdpr_reqs)}: {len(uirementseqR-Related R"   GDPt(f prin  
    )
  PR
   ork.GDewFramComplianceframework=mpliance_co   ts(
     equiremenearch_rct_manager.scontrar_reqs =   gdp
  orkew frammpliancey co barch   # Se  
 ")
  works)}_frameomplianceeq.cn re for cf ijoin(cf.valuorks: {', '.     Framew print(f"  
       ").title}d}: {reqent_iiremrequ{req.f"     •   print(st 2
      ir# Show f:  [:2]ction_reqsata_proter req in d)
    fo"on_reqs)}tectia_pro: {len(datquirementstion Reta Protecatory Dat(f"   Mand  prin    
  rue
    )
ry_only=Tmandato,
        CTIONOTEATA_PRCategory.DmentuireyReqer.Securitct_managy=contracategor        rements(
arch_requier.sentract_manag = coion_reqstect data_pro   ory
categ# Search by   
    s:")
  ents AnalysiequiremSecurity Rn🔍 f"\    print(s
 analysiearch andrements s requiratest Demon  
    #is")
  nalysuirements Aty Reqon("Securiecti   print_s    
 c}")
{i}. {re    "  print(f               ):
s'][:2], 1endationrecommresults['ce_ate(complian enumer i, rec infor            )
tions:"commendaint(f"   Re         prs']:
   endation'recommts[e_resulliancif comp        ndations
Show recomme    #  
          }")
 e']itlq['td']}: {rent_iequireme• {req['r"          print(f          t 3
 ow firs[:3]:  # Shndatoryliant_manon_comp in    for req        )
 ory)}"mandatcompliant_{len(non_ts: quiremenandatory Reiant Mompln-C No  (f"print         
   tory:iant_mandan_complif no              
  ]
  
      mandatory']nd rc['t'] ac['compliant rif no    ']
        liancent_compemeults['requirnce_resin compliaor rc   rc f          ory = [
_mandatcompliantn_   no   
  irementsqudatory rempliant manount non-co# C
            }")
    tions'])_certificas['missingiance_resultin(complns: {', '.joertificatioMissing C     print(f"
          ns']:tificatio'missing_cerresults[mpliance_ coif
               ']}")
 _compliancelts['overalle_resuiancce: {complanl Compliicon} Overalcompliance_  {rint(f"   p       "❌"
lsempliance'] ell_coverae_results['oianc✅" if complcon = "mpliance_i
        co                )
lities']
capabiendor['bilities=vor_capa   vend      ,
   ns']iocatcertifir['ions=vendoficatdor_certi   ven         plate_id,
temud_template.ate_id=clo  templ         nce(
 liaor_compnde_veatger.validt_manarac = cont_resultsnce   complia           
  ame']}")
r['nting: {vendoalidant(f"\n🔍 V   pris:
      test_vendorr vendor in fo 
     E)[0]
 CLOUD_SERVICactType.e(Contrs_by_typlate_temper.getmanagtract_= contemplate oud_ cl    
    ]
}
                }
     nt
  agemeo key man # N: False  01"  "en0           ogging
    audit lNoe,  # ": Fals"al001           
     : True, "cr001"              
 : True,c001"  "b           t
   anagemenrability mulne # No v False, "vm001":          rue,
      ": T   "ir001             MFA
 e,  # No01": Fals"ac0           
     ryptionS encssing TL  # Mie,": Fals"dp002               ": True,
   "dp001            ": {
  bilities   "capa
         onsicatiertifng c,  # Missiype_ii"]_t"soc2 [tions":certifica          ",
  nt Vendor""Non-Complia": "name           {
           },
         }
   ent
      gemey mana: True   # Ken001"        "
        logging  # Audit ": True,   "al001           eporting
   riance # Compl: True, 001" "cr              y
 itontinuss cusine# B1": True,  c00  "b   
           nagementility maulnerab# V": True,  001        "vman
        response plIncident e,  # Tru01": r0       "i   tion
      cahenti autti-factore,  # Mul Truac001":         "it
       n in transa encryptioue,  # Dat: Tr02" "dp0              n at rest
 cryptioen # Data ue,  Trp001":"d           {
      ties":apabili       "c   "],
   "gdpr",, "iso27001pe_ii"c2_tysons": ["ertificatio"c         ndor",
   nt Ve: "Compliaame""n             {
      s = [
 t_vendor    tesmplates
inst tece agaomplian vendor c # Test  ")
    
 idation ValComplianceendor n("Vsectio    print_s")
    
ntequiremecount} rreq_)}: {title(, ' ').place('_'egory.re• {cat(f"        print     0
        tent else  1 if con('\n\n') +ent.countonteq_count = c           r    
 first 3how # S))[:3]:  items(ons.(req_secti list, content inor category       fs:")
     rieategorement Cf"   Requi    print(
        ements']requirrity_s']['secusection'contract_ge[nguaact_lans = contr_sectio      req     ory
 ts by categequiremenow sample r      # Sh      
        
    }")eworks'])liance_fram['compuageng_la(contract{lenorks: mewe Fraomplianc   C"t(frin        p']}")
    ntsuiremeal_req['totuaget_langcontrac {nts:al Requireme"   Totnt(fri      p)
      ame}" {template.n Template:f"  nt(         priully")
    successfgeneratedontract f"   ✅ Cprint(                  
   )
            on']
   tiescrip['service_dariotion=scene_descrip     servic        me'],
   vendor_nao['enarindor_name=sc          veid,
      e.template_atd=templ  template_i           e(
   anguagract_le_contgeneratnager.t_mage = contracguaract_lan        cont 
               ]
tes[0emplate = t    templa
        tes:plaem   if t  
   pe'])plate_tyio['temarype(sceny_ttemplates_bger.get_tract_manas = conatepl
        tem this typeore fmplatte# Get          
    ]}")
   endor_name'{scenario['vract for: contenerating t(f"\n📄 Gin
        prnarios:ario in sce    for scen  
   ]
        }
 atform"
   plnalyticsnd alligence a intesinessBu "on":ptiescri"service_d      ",
      yticsaltaFlow An": "Davendor_name "         ,
  RIPTIONSCUBype.SAAS_StTntrace": Complate_typ        "te
       {        },
    ervices"
 essing sprocta ure and dafrastructoud in "Cl":criptione_desic     "serv",
       sutionloud Sol: "SecureC"endor_name       "vCE,
     LOUD_SERVIactType.Cntrype": Co"template_t          {
     
     ios = [cenar  scenarios
  erent sffr difoct language rate contra# Gene   )
    
 n"ioatage Generct Languon("Contrant_sectipri
    
    version}")ate.templ {n: Versio  f"        print(  ents)}")
requiremance_liompemplate.c cf in tue foroin(cf.val.j: {', 'Compliance   f"    print()
      irements)}"y_requuritmplate.sects: {len(teRequiremen  "       print(fue}")
    ype.val_tntractmplate.coe: {te    Typf" print(        .name}")
 • {templateprint(f"  s:
        emplate in all_t template")
    fors)}):ll_template{len(aTemplates (e bl Availaf"📋rint(    p
  
  ())es.valueser.templatact_managntr= list(co_templates tes
    allplalable temai  # Show av   
  )
 "tesract Templailable Conttion("Avant_sec
    pri    nager()
mplateMaTeyContracturit Secer =nagct_maontra  c 
     
 DEMO")CT TEMPLATESTRACONRITY "SECUr(nt_heade
    pri"t""mengeplate manacontract temte onstra """Dem
   mplates():ontract_te def demo_csync

aentsidn inc retur 
   
   k=True)k(missing_o.unlinath)dence_file_pPath(eviup
    
    # Clean 
   0 days")n the last 3ecorded i incidents rnt(f"   No   pri        
 lse:        e")
i}. {rec}  {nt(f"         pri      
        [:2], 1):ions']ecommendat['rrye(summa in enumerati, rec     for     )
       s:"ommendationt(f"   Rec  prin           ]:
   mendations'comremmary['       if sus
     ionndathow recomme    # S        
   )
         "nt']}k_assessmemmary['rissment: {susesisk As(f"   R print      )
     e']:.1f}%"n_ratsolutiommary']['re'incident_susummary[ate: {solution R Re"       print(f      ")
 incidents']}']['open_nt_summary'incidemmary[s: {sucident  Open In print(f"           nts']}")
 incide]['resolved_y'mmarident_summary['incsunts: {cideInResolved (f"        print
       ]}")s'incidenty']['total_ummart_senincid['ummary: {ss)daycidents (30   Total Inprint(f"           ) > 0:
  , 0nts'total_incidet(' summary.ge   if     
     id, 30)
   endor_ry(vdent_summa_vendor_incinerateacker.geident_trinct ary = awaiumm 
        s   
    ")dor_id}: for {vent Summaryciden📈 Inint(f"\n  pr      ids:
vendor_r_id in ndo   for ve)
    
 nts) i in incideendor_id forst(set(i.v_ids = li    vendors")
    
 Summarieentr Incidndo"Ve_section(inties
    prt summarncidene vendor i# Generat  
    ")
  deadline']}n_ficatio['notiailsetine: {dDeadln tio  Notificaf"     rint(           p     :
        ilse' in detainadln_deficatio if 'noti                   le")
: Applicabr()}.upperk {frameworint(f"        p                ble'):
'applicas.get(ilta deif             ():
   ].itemsis'ance_analyst['complin repor irk, detailsor framewo   f   
      )ons:"e ImplicatincComplia   t(f"rin     p
       analysis']:ance_t['compli if reporons
       e implicatincow complia        # Sh 
      ")
 ions']}ed_actet['complmetrics']report[' Actions: {mpletedint(f"   Co
        prount']}")]['actions_ct['metrics' {reporount:ns CActio"     print(f)
      ount']}"nce_c]['evides't['metricpor Count: {re Evidenceprint(f"  
        us']}")']['statmaryt_sumncideneport['itatus: {r"   Snt(fpri    ]}")
    erity''sev][ent_summary'idt['incorverity: {rep Seprint(f"          tle']}")
['ti']maryt_suminciden {report['  Title:(f"       print  
  d)
      _iincidentrt(incident.nt_repocide_inratener.gedent_trackencit iaireport = aw               
t_id}")
 dent.incideninciport: {Redent ncin📊 I print(f"\      nts
 st 2 inciden fir # Report oidents[:2]: t in incnciden    for iorts
epnt riderate inc  # Gene
    
  porting")and ReAnalytics cident ction("In  print_se
  ")
    gher leveled to hident escalat   ⬆️ Incirint(f":
        pccessalate_su
    if esc )
    
   "y.commpany_analyst@couritecior_s="senescalated_by  
      equired",on rotificatiance nmplilegal and co - edconfirmch "Data brea_reason=lation      escaid,
  cident_nt.inderitical_inci=cent_idincid
        ent(incid.escalate_erent_trackawait incidccess = te_sulasca e  demo)
  for escalationd (simulate ate if neede 5: Escal  # Step
  NG")
    TISTIGAted to: INVE Status updaf"   ✅   print(e:
     atpd if status_u)
    
   
    " of exposure scopeestigatingnv complete, iinmentnitial conta notes="I    ,
   om"any.canalyst@compurity_secnior_ater="se upd     ATING,
  s.INVESTIGtuStaer.Incidentcident_track_status=innew,
        t_idnt.incidenical_incide_id=critincident(
        tatusnt_sncide.update_itrackerincident_ await te =status_updatus
    incident sta4: Update    # Step   
 
  ption}")scri: {action.deleted actionmpf"   ✅ Corint(     p   
    ccess:sue_let if comp   
      )
          
    "any.comomplyst@city_anasenior_secur"ted_by=comple           d",
 entifieomers ided custctzed, affelyess logs anacured, acc endpoint selts="API      resu,
      ction_id.ationtion_id=acac   
         t_id,t.incidenincidenitical__id=crncident   i       n(
  _actiompleteracker.co_t incident awaitte_success =le   compns[0]
     cident.actiol_inritica = con acti       t.actions:
ical_inciden  if crit actions
  almplete initi Step 3: Co
    
    #")ption}descrice.videncription: {ef"      Desprint(     ype}")
   ce_tnce.eviden {evideadded:ce   ✅ Eviden print(f"      idence:
  
    if ev      )

  any.com"complyst@rity_anaby="secuollected_
        cfile_path,ce_=eviden_path file       ",
urel data exposntiapoteing  logs show"API accessn=escriptio  dg",
      e="loypvidence_t        ecident_id,
t.in_incidenalid=criticnt_de       inciidence(
 .add_evrackerincident_t= await    evidence e
    
 e_file.namevidenc= ile_path ce_f     eviden   ")
   ""ied
     tify team noFO] Securit2:18 [IN14:301-15 ed
2024-tely disabloint immediaO] Endp17 [INF5 14:32:-01-1ed
2024 accessntiallys potetomer recordcus 1,247 6 [WARN] 14:32:124-01-15on
20thenticati without aumers exposedt /api/custoinpo API endERROR]32:15 [14:24-01-15 e("""
20_file.writ  evidence
      file:e_encids eve) alse=Faeletg', d'.lofix= suf',ile(mode='wyFmporar.NamedTempfile    with tece file
en sample evid   # Create")
    
 ce...en Adding evidf"   📎  print(idence
  evStep 2: Add # 
    t")
    ity analys securo senior tAssigned✅ print(f"        cess:
   f assign_suc
    i
    com"
    )y.ompannager@ccurity_magner="se        assi.com",
ompanyanalyst@crity_senior_secuee="gnssi
        ant_id,ent.incidetical_incidd=cricident_i    innt(
    cideign_inacker.ass_trait incidentsuccess = awassign_nt
    ncidegn ip 1: Assi
    # Sted}")
    ident_iinct._incidencritical: {denttical Inciaging Cri"\n🔥 Mant(f   prin
     CRITICAL)
entSeverity.ty == Incidif i.severints  incidefor i in next(i cident =inl_criticat
    ical incidene critow with thworkflanagement t mcidentrate in    # Demons")
    
e Workflowdent Responson("Inci print_secti
   }")
    ns)nt.actioincide {len(al Actions: Initit(f"   prin  ")
     l.value}ion_levealatncident.esc Level: {i Escalationf"       print(
   }").valuet.statusciden Status: {inint(f"          pr.value}")
oryegatident.cy: {incegor  Catnt(f"        prir()}")
 lue.uppeeverity.vaincident.srity: {(f"   Seve     printid}")
   .incident_identt ID: {incIncidennt(f"         pri
          ent)
cidppend(ins.aent   incid 
     
        )       ypes']
ected_data_ta['affdatident_ypes=inc_data_taffected            ms'],
teysd_saffecteident_data['s=incected_systemaff          r'],
  reportea['t_datr=inciden  reporte        rity'],
  seve_data['dentciy=initver        se],
    'category'_data[ory=incidentteg        ca    on'],
descriptident_data['nciption=icri       des  ],
   e'['titltaent_dancide=i     titl'],
       'vendor_namedata[t_me=inciden   vendor_na       d'],
  _idata['vendord=incident_or_i        vendident(
    _incer.createtrackt_denait incicident = aw       in
 
        }")le']['titatadent_d: {inci incidentngCreati"\n🚨 int(f     prdata:
   ts_incidennt_data in cide in]
    fordents = [  
    inci
      ]

        }n"]atioformancial_ina", "finersonal_dat": ["p_typesffected_data      "a     ,
 se"]tabadaapi", "["customer_": stemsffected_sy   "a       ,
  y.com"yst@company_analcuritser": " "reporte           .CRITICAL,
ritydentSeve: Inciseverity"    "      EACH,
  ry.DATA_BRentCatego": Incid"category           ,
 t"endpoinured API isconfiged through mexposentially  data potustomertion": "Cescrip"d     ,
       Exposed"Information - Sensitive each ta BrDa"Potential itle": "t        s",
    Solution"QuickFix : ndor_name" "ve           ",
03-RISKY-0": "VENDOR"vendor_id                   {
     },
cs"]
    e_metri", "usagicslyt_anaer ["custom_types":d_dataffecte   "a  ],
       e"inta_pipele", "dagintics_en: ["analyed_systems"    "affect     
   .com",ons@company": "operatiorter    "repM,
        y.MEDIUntSeverit: Incidey""severit            
TION,DISRUPERVICE_ategory.S": IncidentC"category           elines",
 ing pipocessata prting decfailures affmittent ng interrienciexpeics service Analyt: "iption"escr   "d
         on",ce Disruptising Servies Procatae": "D     "titl      ",
 alyticsw AnloaF"Dat": r_name   "vendo         ,
R-SAAS-002""VENDOdor_id":       "ven      
        {
    },
    min_logs"]", "adcredentialser_pes": ["usd_data_ty"affecte        l"],
    in_portaadm"", n_servicethenticatio: ["aued_systems"ect   "aff,
         ompany.com"monitor@c"security_porter": re       "H,
     ity.HIGidentSever Incerity":sev"           ESS,
 ED_ACCNAUTHORIZCategory.UIncident": rycatego "        unts",
   admin accong targetiresses  addicious IPfrom susp attempts ed logine fail": "Multiplescription "d    
       Detected",tempt  Atrized Access": "Unauthoitle"t          s",
  d SolutionloureC"Secu: endor_name"  "v      1",
    OR-CLOUD-00": "VENDvendor_id      "   {
      
     ts_data = [dens
    inciof incidents arious type # Create v
   
    ts")y Incidencuritg Se"Creatintion(sec   print_()
    
 rackerIncidentTer = Vendorent_track
    incid")
    EMENT DEMONT MANAGDEENDOR INCIr("Vnt_heade   pri""
 bilities"gement capant manacideonstrate in"Dem ""
   nagement():_incident_madef demo
async onitor
s_meturn acces     
    r   eted")
plom caintenance Reason: M"  int(f pr    ")
       ssfullysuccevoked ss reAcce"   ✅ nt(f         pri
   cess:f revoke_suc
        i           )
  "
   requireder  no long accesscompleted -intenance son="Ma    rea
        id,ke.grant__to_revo=grantnt_id     gra
       ess(ke_accevotor.rs_moni await accescess =ke_suc      revo
  
        l}")emaier_to_revoke.usor: {grant_ss fng accevoki(f"\n🔒 Reprint
        s[0]rante_gtivoke = acgrant_to_rev   
          ion")
    Revocat"Accessnt_section(    pri
    ve_grants: if action
   ti revocarate access Demonst   #   
 ")
 }. {rec}     {i(f"  print          
    ):ons'][:2], 1ndati['recommertrate(repoec in enume for i, r    ")
       ndations:omme(f"   Rec    print        dations']:
meneport['recom        if rdations
menShow recom#           
 
     1f}%")ate']:.ly_rsis']['anomask_analy{report['rie:  Rat"   Anomaly  print(f
      3f}")_score']:.rage_riskve']['analysisrt['risk_aScore: {repoe Risk Averagint(f"   
        pr.1f}%")cess_rate']:]['sucary'_summ['access {reportess Rate:   Succ print(f"  ]}")
     _activities'ary']['totalccess_summort['arep: {itiestivTotal Acint(f"          pr")
 grants']}active__summary']['['accesss: {reportant  Active Grrint(f"        p days")
 ]}']['days'metadataort['report_eriod: {rep  Report Pprint(f"  
        ays
       Last 7 did, 7)  # r_t(vendos_reporesnerate_acctor.genis_moacces= await    report    
         d}:")
 ndor_i{veeport for  R📋 Accessprint(f"\n      _report:
  ndors_todor_id in veor ven    
    f-002"]
SAASR-"VENDOLOUD-001", VENDOR-C = ["reportdors_to_    venors
or vendts fcess reporate ac# Gener
    )
    Analytics" and rtsess Repoion("Accrint_sect   p  
   ce_ip}")
e IP: {sour  Sourct(f"      in        pr")
    'Normal'}e lsy_detected evity.anomalacti' if edDetecticon} {'omaly_: {an  Anomalyt(f"             prin
     re:.2f}")scok_ty.risivi{actk Score:   Ris"      f     print(
       )}"rce{resou{action} on tatus_icon}    {sf"   print(            
           
lse "✓"tected eanomaly_dety.f activin = "🚨" i anomaly_ico          lse "❌"
 ess eucc "✅" if status_icon =          s     
    )
       
          ess=successucc   s          64)",
   Win64; x0.0;  1 (Windows NTlla/5.0t="Mozi  user_agen            urce_ip,
  ce_ip=sosour        e,
        urce=resourc   reso            n,
 actio  action=   
           _token,t.accessken=gran       to     y(
    _activitg_accessr.lonitoit access_mo= awa activity            tivities:
ss in acceip, suc, source_ resourceon,r acti
        fo
        ivities:") access actimulatingnt(f"\n📊 S  pri
               ]
  
     usspicio + suailede)  # F1", Fals"203.0.113.a", ustomer_dat, "c_export"   ("bulk    ),
     0", True68.1.10"192.1", a "user_datin_access",  ("adm          ious IP
picrue),  # Sus.0.50", T"10.0s", "log_filegs", delete_lo      ("
      True),1.100",  "192.168.files",config_"", tingste_setupda"          ( True),
  0",8.1.10192.16, "tabase""daig", onf   ("read_c        es = [
   activiti
      tiesctiviccess a ate various# Simula     
         ")
  %M:%S')}H:('%Y-%m-%d %metrfti_at.spires{grant.exes: pir"   Exint(f       prvalue}")
 s_type.t.accesangress Type: {"   Accrint(f    p
    ant_id}")nt.gr {graGrant ID:(f"   int        pr")
ail}nt.user_emgraor: { fng accessitori Monn👤\nt(f"    pri  rants[0]
  tive_g = acgrant       nts:
 grave_
    if acti
    ts()_active_granget_monitor. accesse_grants =iv
    actctivitiessimulate ats and an grctiveet a  
    # G
  itoring")tivity Mon Acession("Accprint_sect
    )
    ed}"ir_requ.approverest2qut access_re {noed:ov-appr Auto     print(f"
 y_access}")rgenc2.emess_requestcecess: {acEmergency Ac   int(f"}")
    prype.valuecess_t_request2.acccesspe: {a Tycess(f"   Ac)
    printlue}"tatus.va_request2.sss{acce"   Status:     print(fd}")
uest_iequest2.reqs_rces{acequest ID: int(f"   R
    pr
    True
    )rgency= eme1,
       hours=duration_
        ",ss requiredediate accemm- istem outage ical syon="Critificatiustss_j  busine],
      user_data"tabase", "roduction_das=["purce     resoIN,
   e.ADMcessTypcess_type=Ac      acw.com",
  rt@dataflouppoer_email="s     request-002",
   NDOR-SAAS"VEvendor_id=  
      (quest_accessnitor.remot access_uest2 = awaieq   access_r
 t")
    esccess Requmergency Aio 2: En🚨 Scenarint(f"\    prss request
cce ancy 2: Emerge# Scenario    
    nager")
rity maed by secuccess approv  ✅ A print(f"  )
              "
com@company.agersecurity_manapprover="        id,
    quest_request1.ret_id=access_ues        req  
  request(rove_access_appess_monitor.ait accsuccess = awoval_      appred:
  r_requirrovequest1.appf access_rered
    iquirove if re   # App
 
    s}")ency_accesrg.emet1esccess_requ {ass:gency AcceEmer"       print(fquired}")
prover_reest1.apqureed: {access_ Requir  Approvalf" 
    print(e.value}")cess_typest1.acccess_requss Type: {ace(f"   Ac)
    printalue}".vusequest1.stat{access_rtus: Starint(f"   d}")
    pest_iquest1.requss_rest ID: {acce Reque print(f"  )
    
      e
 alsemergency=F        =4,
urson_hodurati        ,
pdates"figuration u and conintenance ma"Monthlycation=stifis_ju  busines
      "],ig_files", "conf["databaseresources=      WRITE,
  Type.READ__type=Access access       
ud.com",clouren@secemail="admiuester_     req",
   -001LOUD-Cid="VENDOR vendor_
       ccess(.request_aonitorss_m accewaitquest1 = aaccess_re
    
    e Access")Maintenancard o 1: Stand🔐 Scenari"\n   print(f request
 d accesstandar1: S Scenario     
    #rios")
 ScenaRequestss Vendor Acceion(" print_sect
    
   ssMonitor()AccendorVe_monitor = ess  acc    
  O")
ITORING DEMCESS MON("VENDOR ACderprint_hea   """
 ieslitg capabionitorinss mccedor aonstrate ven"Dem"():
    "ringss_monitovendor_accec def demo_yn

asrn sboms  
    retu
  ue)=Trk(missing_okame).unlinemp_file.nth(t        Pa       eanup
      # Cl             
              )
es"yte_size} b {file:e siz     Filprint(f"                    
 st_sizestat().ile.name).th(temp_fe = Pa   file_siz              
   ")atform.value} typeto {format_orted   ✅ Expf"       print(             :
 port_success   if ex            
              )
           
        .nametemp_filetput_path=        ou         pe,
   t=format_ty       forma        
     .sbom_id,sbom_id=sbom                _sbom(
    .exportgeranait sbom_mcess = awaport_suc     ex          _file:
 as templete=False)  deype.value}',_t{formatx=f'. suffi='w',e(modeilemporaryFNamedT tempfile.    with  ats:
      type in formor format_    f     
    ON]
   E_DX_JSCYCLONSBOMFormat.SON, X_Jmat.SPDSBOMFor = [   formats    [0]
 ms sbom = sbo
       
        ilities")port Capabon("SBOM Ext_secti        prins:
sbomf rmats
    iifferent fort SBOM in dExpo  # 
  ")
    onents']}compfied_']['modimmarymparison['sunents: {compoed Coodifirint(f"   M        pts']}")
mponenoved_co['remsummary']on['comparismponents: { Removed Corint(f"        p  nts']}")
d_componemary']['addeson['sum{comparinents:  Compoded Ad  (f"     print")
   nents)om2']} composbnents__compotalmmary']['to'suon[omparisre_name} ({cwasoft].boms[1 SBOM 2: {s print(f"  )
       nents)"]} compos_sbom1'ponentom']['total_cn['summaryparisome} ({comare_na[0].softw 1: {sboms"   SBOM(frint
        pd)
        _isbomoms[1].bom_id, sb.s[0]_sboms(sbomsare.compom_managerawait sbarison = mp       coSBOMs:")
 paring f"\n🔄 Com  print(2:
       >= (sboms)
    if leniple multif we haveare SBOMs # Comp
    
    risk']}")['license_']ntssmesek_asalytics['ris {anse Risk:icenrint(f"   L    p    sk']}")
lity_ri['vulnerabient']smrisk_assesics['analyt Risk: {rabilityf"   Vulnerint(       p)
 censes']}"ue_liniqsummary']['ulytics['ses: {anaue Licent(f"   Uniq  prin)
      es']}"erabilitil_vulntota']['summarytics['{analy: ilitiestal Vulnerab To print(f"         ts']}")
ene_compon'vulnerablummary'][cs['sanalyti: {onentsnerable Comp Vul"  print(f    )
    s']}"enttotal_compon['mmary']sulytics['na {ats: Componenf"   Totalt(   prin          
  id)
 bom_.ss(sbomtic_analyomager.get_sbbom_mancs = await slyti      ana
        :")
  tware_name}r {sbom.soffolytics Anant(f"\n📊 pri    :
    m in sbomsr sbo   fo SBOM
 chr eas fotice analy# Generat   
    
 on") Comparis andalytics("SBOM Anrint_section  
    p")
  ount}title()}: {cype.   {comp_t"      print(f  
          ms():_types.iteponentunt in comtype, cofor comp_         es:")
   omponent Typ(f"   Crint     p 
               1
    ype, 0) +get(comp_tpes.component_tye] = yp[comp_ttypescomponent_             e.value
   yp.component_tonentype = comp_t     comp        
   ts:mponenn sbom.conent ifor compo         
   types = {}nent_compo           eakdown
 nt bromponeow c Sh          #    
        ")
  nships)}atiorel(sbom. {lens:elationshipt(f"   R    prin        
ents)}")onsbom.comp {len(onents:  Compf" print(    ")
        t.value} {sbom.formaat:ormf"   F      print(      }")
.sbom_idM ID: {sbom  SBO" t(f prin           ")
cessfullyerated suc  ✅ SBOM gen print(f"             
           sbom)
pend(    sboms.ap
                )
              X_JSON
  SPDat.FormOMSBom_format=      sb    ],
      on'['version=projecttware_versi         sof'],
       ject['name_name=pro    software          p_path),
  th=str(tem_pare  softwa            '],
  r_idvendoect['id=projdor_ven              om(
  te_sbnerager.ge_mana await sbomom = sb     OM
      rate SB  # Gene        
             ]))
 ements_txt''requirect[roj'\n'.join(prite(    f.w            as f:
    'w') e, n(req_filth ope        wi    "
    rements.txtuith / "reqmp_pa = te_file req            :
   ectxt' in projnts_tf 'requireme    i        
            indent=2)
f, e_json'], ect['packagn.dump(proj         jso        
   as f:file, 'w') ckage_n(paope  with             json"
  / "package.path  temp_e_file =kagpac             ct:
    in projee_json'ckag   if 'pa  les
       fidependency  # Create                
 )
       emp_dirath = Path(tp_p         tem:
   temp_dirs ectory() aemporaryDirempfile.Tth t
        wi structureary projectemporte t     # Crea
   )
        version']}"oject['} v{prame']ect['nor: {projg SBOM fratinf"\n📦 Gene print(     ts:
  ojecsample_prt in projecfor 
    = [] sboms )
    
   Materials" Bills of g Softwareratinon("Genet_secti prin
    
      ]}
        ]
            19"
 .0.lchemy==2"sqla          
      ",.2=2.3flask=      "   ",
       .031s==2.   "request          3",
   ==1.24.umpy   "n             =2.0.3",
 "pandas=       [
        : t"ments_tx"require           
 AS-002",R-SAd": "VENDOendor_i         "v0",
   .5.sion": "1"ver         ssor",
   DataProcee": " "nam                {
   },
                }
   }
               5.88.0"
  : "^ "webpack"                
   0","^29.5. "jest":             
       ies": {ndenc   "devDepe                },
        
     .4""^2.29moment":  "                   0",
": "^1.4.    "axios               ",
 1": "^4.17.2odash"l                 18.2",
   "^4. xpress":       "e          ",
   .2.0 "^18t":      "reac       
       ": {endencies"dep               
 .2.1",on": "3ersi       "v,
         webapp"e": " "nam             : {
  ckage_json""pa          ",
  OUD-001CLR-"VENDO": _id  "vendor
          .1",: "3.2sion"       "ver",
     ebAppe": "W "nam          
 
        {cts = [sample_projetion
    generaOM ware for SBsoft sample te   # Crea    
 )
er( SBOMManag =er  sbom_manag   
  DEMO")
 LS (SBOM) TERIAMA BILL OF "SOFTWAREt_header(   prin"
 ities""abilcapnt OM managemeate SB"Demonstr
    ""ement():anagemo_sbom_m ddefync 

astsuln_resturn sca  
    re}")
  endations'])ecommrt['r(repoovided: {lendations pren  Recommrint(f"  p")
       neratedty report geiled securi"   Detat(f    prin    (riskiest)
can_reportrate_snet scanner.geawaireport =      
      e}")
     e_namiest.software: {riskagt Risk Pack🚨 Highest(f"\nprin        e)
cor_risk_srallr: sr.oveey=lambda slts, kax(scan_resukiest = m    ris:
    esultsn_rif scackage
    risky past  moport foretailed rerate d # Gene
    
   tected}")e_de: {malwarres with Malwa   Package print(f"")
   ors}otal_backdo: {tcators Indial BackdoorTot"    print(fs}")
   ievulnerabilit {total_rabilities: Vulneotal   Tf"  print(
  ")tal_scans}canned: {tokages SPacTotal "      print(fy:")
 n Summarnt(f"📊 Sca 
    pricted)
   re_detewasr.mallts if su in scan_refor srted = sum(1 detec   malware_sults)
 can_rer in sators) for sr_indicbackdoon(sr.um(ledoors = sl_backotalts)
    t_resur in scanfor silities) vulnerabn(sr. = sum(lebilitiestal_vulneralts)
    to(scan_resucans = len  total_s)
    
  Summary"ity Scan ilerab("Vuln_section
    print    ue)
ng_ok=Trunlink(missiath).temp_file_pPath(          anup
   # Cle          
 ly:final             
  ")
     el}sk_levcator.ri{indik:  Risdence:.2f},nfiator.co{indiconfidence:        Cnt(f" pri                
   ")}tion.descripator}: {indiccator_typedicator.indi- {in"     t(fprin                    t 3
Show firs3]:  # ors[:oor_indicatkd.baccan_result in sndicator i    for            ators:")
 IndicoorckdBa ⚠️  (f"  rint       p
         indicators:backdoor_ult.n_ressca   if        
  catorsr indioockd# Show ba       
               ")
  {vuln.score}ore: value}, Scn.severity.verity: {vul   Sef"       print(                ")
 cription}uln.des}: {ve_id.cv     - {vulnrint(f"        p            irst 3
how f Sties[:3]:  #libiult.vulnera_resn scan  for vuln i           es:")
   itibilVulnera   🚨 print(f"            ities:
    vulnerabil_result.can        if setails
    nerability dhow vul       # S
             
    }")iles.scanned_fscan_resulted: {Scann"   Files int(f         pr}")
   d else 'No'teare_deteclt.malwan_resuif sc{'Yes' d: Detectelware   Maf" print(            )
s)}"catorr_indickdooan_result.basc: {len(rsr Indicatodooack   Bprint(f"           ies)}")
 ulnerabilitult.v(scan_res {lenound:lities Fulnerabi(f"   V    print  ")
      re:.2f}/10risk_scorall_lt.ovecan_resu{sRisk Score: Overall   int(f"           pr")
  secondson:.2f} atiult.scan_duran_resn {scpleted icomScan nt(f"   ✅      pri
           
        sult)d(scan_re.appentsscan_resul                 
)
                   ]
ersion'ge['version=packare_voftwa         s       'name'],
package[e_name= softwar          h,
     path=temp_file_oftware_pat         s
       _id'],ndorve['ackage=p vendor_id       e(
        _packag_softwareaner.scann= await scsult    scan_re       
  try:  
        e
      le.namh = temp_fiatmp_file_p       te     )
']e['contente(packagle.writ temp_fi           p_file:
 as temlete=False)='.py', de', suffix(mode='wlearyFiemporpfile.NamedT   with temfile
     y ate temporar Cre       #        
 ]}")
ion'ackage['vers} v{p']ckage['nameng: {pan🔍 Scanniprint(f"\       ackages:
 e in test_p packag for= []
   results   scan_  
  ages")
  Packre twaing Sofn("Scann_sectio
    print     ]
  
 
        } """           n123"
= "admid n_passwor   admi       ls
  redentiaed c  # Hardcod          
          
  ")gerous')entially danotl("print('p       eva     n
xecutioic code enamDy#                    
     WQ=")
X3BheWxvYaWNpb3VzFsdecode("bW64 = base64.b payload   d
         payloancodedus base64 e  # Suspicio          
        t eval
       impor
         mport base64      i  
    ": """  "content        03",
  -0-RISKY": "VENDORendor_id  "v       
   ",0.0": "1.n   "versio,
         ableLib"ner: "Vulname"  "                 {
 },
    ""
          "
            }      }
      
          2.29.0""^ent":        "mom            
 ^4.17.21",sh": "    "loda           ",
     "^4.18.0: ress"     "exp             
  dencies": {"depen               2.1.0",
 n": "versio        "
        p",eApecur": "S  "name         
      {          """
 nt": conte      "",
      -001LOUD"VENDOR-C: or_id"vend   "
         "2.1.0",sion": ver  "          
cureApp",ame": "Se         "n   
        {
es = [kagtest_pac   r testing
 ges foe packasoftwarte sample     # Crea 
  canner()
 tySoftwareShirdPar scanner = T
    
   ")G DEMOSCANNINTY RABILI("VULNErint_header
    pies"""litbicanning capality snerabitrate vulonsem
    """Dning():ansclity_rabinef demo_vulsync de

aessmentsn assretur   
    
 )}")-%d'ime('%Y-%m.strftview_daterext_isk.neest_rduled: {highreview schet    Nex(f"
    print findings")ndings'])}ort['fiep(ren{l with neratedport ged rele Detaiprint(f"  sk)
    t_rihighesent_report(ssessmate_aessor.genert assawaieport = 
    
    r")_id}risk.vendor: {highest_endorisk V🚨 Highest Rnt(f"\nrire)
    p.risk_sco a: aambdants, key=lx(assessmek = mat_ris
    highesvendorest risk ort for highed repdetailShow   
    # ")
  (s)unt} vendoritle()}: {coevel.trisk_lf"     {t(     prinems():
   ribution.it risk_distcount invel, or risk_le:")
    fributionRisk Dist(f"     print)
  2f}/10"k_score:.vg_risScore: {aerage Risk   Av"     print(fments}")
sstotal_assesessed: { Vendors Asotal(f"   Tnt
    priry:")t Summa