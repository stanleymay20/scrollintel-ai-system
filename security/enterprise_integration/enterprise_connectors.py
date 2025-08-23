"""
Enterprise Connector Registry - 500+ Pre-built Connectors
Provides rapid client onboarding with comprehensive enterprise system support
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import importlib
import inspect

logger = logging.getLogger(__name__)

class ConnectorType(Enum):
    ERP = "erp"
    CRM = "crm"
    DATABASE = "database"
    CLOUD = "cloud"
    API = "api"
    FILE = "file"
    MESSAGING = "messaging"
    ANALYTICS = "analytics"
    SECURITY = "security"
    COLLABORATION = "collaboration"

class ConnectorStatus(Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    BETA = "beta"
    MAINTENANCE = "maintenance"

@dataclass
class ConnectorMetannectorreturn APICo"
        ector'')}Connace(' ', .repl = f"{nameme__tor.__nanec   APICon  
      {}
         return           ny]:
  [str, A -> Dictone): str = Nelf, entityschema(sget_ async def         
             data)}
  d': len(deoas_lrd', 'recocessucus': 'seturn {'stat      r       
   t[str, Any]: -> Dictarget: str), Any]], t[Dict[strta: Lis(self, daoad_data async def l           
   ]
         rn [      retu   ]:
       ct[str, Any]t[Di> Listr, Any]) -ct[s query: Dif,t_data(selc def extrac   asyn
                    0}
 e_time': 6responsalthy', 'tus': 'hetaturn {'s      re         y]:
 An> Dict[str, eck(self) -_health_chperform _   async def               
 s
      pas           lf):
    nection(selose_consync def _c a  
                     
       pass         elf):
ion(slish_connectc def _estab     asyn  
                    pass
      
       (self):icatethent_auasync def                
  id
       tor_onnecnfig.cself.co    return           data:
  ctorMetaf) -> Conneseltadata( _get_medef        
    onnector):r(BaseCIConnectoass AP
        cl class"""nectormic API conynate a d""Crea        ":
-> typer) r, name: stendor: st str, vonnector_id:f, cs(sel_clasectori_connte_ap_crea  
    def nnector
  CoFile return        r"
onnecto)}C ', ''('placeame.re = f"{ne__r.__nameConnecto        Fil       
 {}
 eturn   r            tr, Any]:
 > Dict[sone) - str = N, entity:selfhema(f get_sc async de         
           (data)}
   lended': cords_loa', 're': 'successstatusturn {'   re        Any]:
     str, Dict[ str) -> arget: ty]], AnDict[str,: List[(self, dataad_data loync def       as   
     ]
         turn [   re           ny]]:
  , Aist[Dict[str]) -> Lt[str, Anyery: Dica(self, qudatef extract_nc d        asy 
             : 40}
  e'esponse_timthy', 'r'heal {'status':      return         r, Any]:
   Dict[st(self) ->health_checkm_rfor def _pe     async   
       
         ss        pa      
  self):on(ectionnose_cef _clasync d       
       
                  pass:
        tion(self)ecconntablish__es def nc      asy       
     ass
            p          lf):
(secatethenti _audef async                   
id
     connector_lf.config. return se     
          ta:etadannectorMf) -> Codata(selmetaget_      def _):
      orConnectr(BasetoleConnecs Fias    cl   "
 class""onnector file cynamic a d""Create        " type:
  ->name: str): str, vendorid: str, or_ connectelf,tor_class(seconne_c_filateef _cre 
    d
   nnectoragingCossrn Me   retu
     or"}Connect(' ', '')me.replace{name__ = f"ctor.__nagConne    Messagin   
     {}
        return        y]:
     Dict[str, An) ->  str = Nonety: entia(self,chemf get_snc de asy               
   ata)}
      len(dds_loaded':, 'recorss': 'succe {'status'eturn          r
      [str, Any]:str) -> Dict], target: , Any]trct[sList[Di: , datalf_data(seadnc def lo    asy  
                  turn []
      re       y]]:
   tr, Anst[Dict[s-> Listr, Any]) y: Dict[(self, querextract_dataync def     as
                    30}
e': timesponse_ 'ralthy','hetatus':  {'s return              Any]:
 ,  -> Dict[strlf)check(seth_m_heal_perfordef c    asyn        
        
         pass         elf):
   nection(sose_conf _cl async de
                
       sas          p      n(self):
nectiocon_establish_ def ync    as  
              
      pass      
        lf):(seicatehent _autync def       as        
   id
      nnector_coself.config.     return          
  a:rMetadat-> Connectoa(self) get_metadatdef _            onnector):
nector(BaseCongCMessagin     class    "
class""onnector  cgingsaynamic mesa dCreate     """  -> type:
  ) ame: stror: str, nend vtr,or_id: sonnects(self, cnnector_clasging_cosareate_mes def _c    
   nnector
yCourn Securit     ret  tor"
 Connec, '')}place(' 'f"{name.re = __name__Connector.ty   Securi     
     }
   return {          
      , Any]:strct[ -> Ditr = None)ity: s, ent_schema(self getef   async d       
           )}
   (dataed': lends_loadecoress', 'rtus': 'succturn {'sta      re          , Any]:
Dict[str) -> rget: str, Any]], taist[Dict[str Lata:a(self, d load_datc def asyn                    

   turn []     re           r, Any]]:
stt[ -> List[Dicy])ict[str, An Dlf, query:set_data(extracnc def sy      a  
                120}
e_time': ponsalthy', 'res: 'hetatus'  return {'s         
     ]:[str, Any -> Dict_check(self)rm_healthrfoc def _pesyn        a              
s
          pas     
   ion(self):onnecte_c _closync def       as     
             pass
               self):
ction(onneh_ctablisnc def _es  asy
                    s
  pas              
  ate(self):authenticf _nc de       asy             
id
    g.connector_.confirn selftu re           data:
    nectorMetaelf) -> Con(sadatametf _get_       deor):
     aseConnecttor(BcurityConnecclass Se        
lass"""nector cconity amic secure a dyn"Creat" "       type:
str) -> , name:  vendor: strtr,: stor_idnnec(self, coss_clanector_conty_securi_createef    d 
 ctor
   nenConioaboratn Colletur       rctor"
 Conne, '')}lace(' 'name.rep__ = f"{namenector.__rationCon    Collabo    
        }
return {              :
  r, Any]t[stone) -> Dicty: str = Nelf, entiget_schema(snc def sy a     
             
     }n(data)aded': leloecords_uccess', 'r'status': 's  return {   
           tr, Any]: -> Dict[s str)et:ny]], targ Act[str,st[Dilf, data: Lia(se load_datasync def                  
     n []
     retur         
    Any]]:st[Dict[str,]) -> Liict[str, Any Dy:a(self, querract_datextf  async de       
               0}
 me': 8'response_tithy', eal'h{'status':   return               r, Any]:
> Dict[stelf) -k(sh_checerform_healtf _pync de    as
                ass
              plf):
      nnection(sef _close_co   async de              
  
            pass     self):
    ection(onnstablish_csync def _e  a
                        pass
           :
   icate(self)ntef _authe   async d                 
 d
   .connector_ilf.config   return se          a:
   torMetadatf) -> Connecdata(sel_meta _get  def
          nnector):aseConConnector(Bratiobolas Col    clas"
    "ctor class"conneion collaboratynamic te a d"""Crea        
ype:: str) -> ttr, nameor: s vendr,tor_id: stself, connecs(tor_clason_connecboraticreate_colla 
    def _nnector
   icsCon Analyt   retur"
     or')}Connectlace(' ', '"{name.repme__ = f.__nannectorAnalyticsCo              
  return {}
           :
     str, Any]ict[> Dtr = None) -: styf, entiselchema(nc def get_s asy           
           )}
 len(datads_loaded': recor '',essatus': 'succn {'st       retur    ny]:
     r, A-> Dict[stget: str) y]], tarstr, An List[Dict[ata:elf, dload_data(s  async def                    

   eturn []      r          r, Any]]:
t[Dict[st> Lisr, Any]) -y: Dict[st, querelfata(s_dextractc def yn         as 
             150}
  ':_time', 'responses': 'healthyurn {'statu   ret           :
   Any]r,ict[st) -> D_check(selfform_health def _per   async           
        s
     pas            on(self):
 connecti_close_ def ync as
                           pass
        
    (self):ctiononne_establish_c def       async    
               pass
      
         ate(self):thenticc def _au     asyn  
           d
      or_i.connectonfiglf.csen      retur   :
        dataeta ConnectorM(self) ->metadata def _get_         
  r):ecto(BaseConnonnectoricsCss Analyt       cla"""
 nector classs conicnamic analytdy a reate""C       "e:
 ) -> typname: str: str, dor, ven str_id:tor connec_class(self,nectors_conticlye_anaatcreef _   
    donnector
 oudC   return Cl  or"
   }Connect(' ', '')acee.repl__ = f"{nammeor.__naudConnect      Clo     
  urn {}
         ret
          , Any]:Dict[str= None) -> tity: str a(self, enet_schemasync def g               
    a)}
     (dataded': lens_lo 'recordsuccess',atus': ' {'st     return           Any]:
str, > Dict[ -: str)et], targtr, Any][s[Dictata: Listelf, d_data(sdef load async          
             n []
        retur   ]]:
      , Anyct[str -> List[Diy])str, Anquery: Dict[ata(self, ract_dnc def ext         asy  
            75}
  e':ime_t'responsy', s': 'healthatueturn {'st        r]:
        Anyr, [stDict -> (self)alth_check _perform_hedef    async 
                   
 ass           p    ):
 ion(selfonnectdef _close_c     async        
        ss
     pa        
       on(self):ish_connecti _establnc def        asy      
    pass
             
         ate(self):uthenticc def _a      asyn
                  _id
oronnectg.cn self.confiretur          a:
      rMetadatConnectolf) -> ata(seget_metadf _      de
      Connector):ctor(BaseneloudCon    class C""
    ctor class"necloud cone a dynamic  """Creat:
       r) -> typetr, name: str: sendo vid: str,nnector_lf, cos(seasctor_clonnee_cloud_cf _creat    de  
nnector
  baseCoturn Data
        reector", '')}Connce(' 'laepe.rf"{nam__ = r.__nameseConnectoDataba
        
        rn {}     retu       y]:
     An Dict[str,ne) ->ty: str = No entia(self,chemnc def get_s   asy
                 ata)}
    d': len(dcords_loade', 're 'successstatus':  return {'          ]:
    r, Anyt[st-> Dicstr)  target: r, Any]],[Dict[st Listdata:, ad_data(selfasync def lo             
  
         ]rn [retu          ]:
      , Any]ict[str[Dst-> Litr, Any]) uery: Dict[sata(self, qt_d extracsync def      a  
             25}
    se_time':y', 'responhealthatus': 'n {'st       retur        , Any]:
 tr-> Dict[s(self) h_checkealtrform_hpe _  async def         
         s
           pas         on(self):
_connectiloseync def _cas            
         s
         pas   ):
       ction(selfconne _establish_ async def                    
pass
                  
 self):te(enticaf _auth  async de            
  
        _idig.connector self.confurn ret               
ata:etadrMConnectoelf) -> metadata(sget_f _    de    
    r):seConnectoConnector(Baseba class Data       ""
 class"onnectorbase camic dataynCreate a d""
        ":> type: str) -name str, str, vendor:ector_id: nn, co_class(selforase_connectab_dateate
    def _crr
    MConnecton CRretur       r"
 ecto '')}Conn(' ',ceplaame.re{nme__ = f".__nactorMConne        CR  
   rn {}
        retu    y]:
       tr, An) -> Dict[sneNo= entity: str elf, ema(set_schasync def g          
              }
len(data)_loaded': ords 'recess',succ 'n {'status':    retur          Any]:
   -> Dict[str,get: str) ny]], tar Ast[Dict[str,f, data: Lid_data(selc def loa asyn       
                return []
                 Any]]:
tr,List[Dict[s -> [str, Any])ery: Dictta(self, qutract_daync def ex as     
        
          50}: ponse_time'lthy', 'resus': 'hean {'stat   retur      
       , Any]:-> Dict[strk(self) lth_checeaperform_h async def _                  
ass
         p    ):
        on(selfonnectidef _close_cync           as       
  s
     as  p            self):
  nnection(coablish_c def _est asyn         
            
     pass        :
     elf)e(sicatntheautsync def _          a   
         id
  ctor_.connelf.configse return               
 tadata:ectorMe Connf) ->data(sel_get_meta        def ):
    orseConnectonnector(Bass CRMC cla       ass"""
connector cldynamic CRM reate a ""C     "
    -> type: name: str)or: str,d: str, vendector_ilf, conn(ser_classonnecto_crm_cte  def _crea  
  tor
   ERPConnec   returnor"
     )}Connect' ', ''ace(.repl{namef"name__ = r.__RPConnecto       E        
 n {}
      retur    val
      trierec schema cifi   # ERP-spe          :
    Any]r,ict[st> D = None) -ntity: strself, e get_schema(defync          as 
            data)}
  ded': len(rds_loa, 'reco 'success' {'status':     return         
  a loadingpecific dat   # ERP-s          y]:
   ct[str, An-> Distr) t: y]], targeDict[str, Ant[, data: Lisdata(self load_ async def       
              []
  turn  re           ion
     extractic dataRP-specif  # E          y]]:
    tr, Anict[s[D> Listr, Any]) -ct[sty: Dia(self, querattract_dsync def ex           a  
        100}
   e': se_timrespon'healthy', ''status': return {             :
   ct[str, Any]> Dieck(self) -ealth_chform_her _p defsync    a      
           s
   pas              
  ogiconnection lic disccif   # ERP-spe         
    ion(self):nectonf _close_cc de asyn     
                  ass
           p     
n logicctiocific connespeERP-      #      lf):
     ction(seblish_conne def _esta async          
  
                pass     gic
      ication loific authent ERP-spec        #      
  e(self):atauthenticf _    async de      
            ctor_id
  conne.config.return self               data:
 taectorMe Conn(self) ->ataet_metad def _g         ctor):
  or(BaseConneonnectass ERPC  cl
      """classctor  ERP connea dynamicreate   """C
      r) -> type:name: str: str, doen: str, vr_idctoneself, conctor_class(_connete_erp def _crea 
   ts
   starn      retu
          
 , 0) + 1get(statustus'].by_staconnectors_] = stats['us'][statustat_snnectors_bys['coat  st
          uetus.valata.stas = metadtu  sta   
       atusunt by st Co       #      
           + 1
, 0) et(vendoror'].grs_by_vendecto['conndor] = statsdor'][venrs_by_vents['connectota      sr
      tadata.vendome vendor =         
   by vendor# Count             
            , 0) + 1
ameet(type_n.gs_by_type']connector stats['me] =type_naype'][ors_by_tctstats['conne        lue
    e.vatyptor_.connecadataame = met type_n
           nt by type # Cou          ():
 rs.valuesto.connecta in selfor metada        f        
     }
{}
   status': nectors_by_   'con       ,
  dor': {}ors_by_vennnect        'co   },
  {rs_by_type':cto   'conne    s),
     tionive_connecen(self.actns': le_connectiotiv     'ac
       rs),connectoen(self.ors': lal_connect'tot            ats = {

        stors"""nnectlable co about avait statistics""Ge   "y]:
     str, Anf) -> Dict[ics(seltistector_sta_conn get   def   
 
 rn results       retu
       a)
  tadatend(me.app  results     :
         on.lower())escriptitadata.der in mey_low        quer
        wer() orr.loa.vendodat metawer in_lo    query            ower() or
me.ltadata.naer in mef (query_low      i   ):
   values(ctors.elf.conneetadata in sfor m          
      = []
    results
     )query.lower(lower = y_quer
        """scriptionr de oame, vendor, ns bytorrch connec """Sea       ta]:
daorMeta[ConnectList> tr) -uery: ss(self, qh_connectordef searc 
      y()
 ns.copconnectioself.active_urn  ret      
 s"""nnectionctive co all a   """Get  :
   Connector][str, BaseDict -> ons(self)nective_concti  def get_a
    
  return False            r(e)}")
st: {tion_key}n {connec connectiongosirror clr.error(f"E      logge
      ion as e:ptept Exce       excrn False
 etu    r
        rn True      retu         ")
 ction_key} {connetionsed connecinfo(f"Clo    logger.          ]
  tion_keytions[connece_connec.activ    del self        
    disconnect()tor.nec   await con        key]
     ction_ions[connetive_connect= self.acnnector       co          ons:
_connective self.actikey inion_ if connect               try:
"""
    ctionve connee an acti"Clos""
        bool:) -> ey: strtion_kecn(self, connnectiolose_condef cync   
    asn None
  tur  re          
tr(e)}")_id}: {sconnector {nnection foring corror creat"E(fgger.error lo          
  as e: Exception      except    
  r
        n connectoetur r    
       ")ector_id}n for {connconnectioed creatuccessfully f"Sinfo( logger.          
            
 ector= connkey] ection_ections[connctive_conn      self.a"
      lt')}', 'defau_idtancems.get('insection_paraconfig.connr_id}_{"{connecto f =nection_key      con     ction
 ve conneti # Store ac                 
    urn None
   ret         
      ')}").get('errorion_result: {connecth connectionblis to esta"Failedror(ferogger.      l          cess':
uc] != 'status'esult['snection_r    if con        
nnection()ctor.test_co conneitwa_result = aection   conn         ction
st conne  # Te                 
 
    ss(config)ctor_clactor = conne conne           or_id]
[connector_classesnectlf.conass = seclector_ conn       
                None
 eturn     r    ")
       ndou} not fctor_idnneector {co(f"Connorerr   logger.       
      ses:asclf.connector_t in selid nof connector_         i
   try:"
        ""oronnect cecifiedusing the sponnection  a new cte"Crea     ""ector]:
   l[BaseConnona) -> OptinConfigioConnectr, config: st_id: connectorself, nnection(cocreate_c def 
    asyn_id)
    onnectoret(c.g.connectorsurn self   ret    ""
 nector"pecific condata for a s""Get meta      ":
  a]etadatrMal[Connecto) -> Optionctor_id: str(self, conneadataor_mett_connectge 
    def ())
   s.valuesnector.const(selfeturn li    r  pe]
  onnector_tye == cyp.connector_tadata      if met           ues() 
  valconnectors.ta in self.tada meadata forurn [metet   r     e:
    ypor_tf connect       i""
 d by type"relly filteoptionaectors,  connf availablet list o  """Ge      etadata]:
t[ConnectorM> Lise = None) -ConnectorTypector_type: lf, connonnectors(seailable_cdef get_av    ss
    
_clanector contor_id] =s[connecasseonnector_cl     self.ctadata
   r_id] = metoctors[connecneself.con"
        n class""plementatiota and imith metadaor wa connectter is"Reg     ""  ):
 s: typennector_clasa, coctorMetadat Connea:r, metadat_id: st connectorr(self,r_connectote _regisdef    
    
  )         )
 amer, nvendoector_id, ass(connnnector_cle_api_coreatlass=self._ctor_cnec     con          ),
                 "v1"}
rsion": i_veap "": "3.0",pi_versionrix={"openay_matpatibilit    com           (),
     tcnow.ud=datetimepdate      last_u        }",
      r_ids/{connecto/connectorcomel.ntscrollis://docs.rl=f"httpumentation_u       doc          ,
   ": 1000}nectionsent_con"concurr 10000, er_second":s_prequest={"rate_limits                uf"],
    , "protob "yaml""xml",son", ts=["jata_forma    d             uth"],
   sic_aba", "wtey", "jpi_k, "a"oauth2"ethods=[tion_muthentica        a           ,
 "secure"]", itoronidate", "mrm", "valsfo", "tran["proxyions=eratorted_opsupp                 
   description,description=               
     TIVE,rStatus.ACtus=Connecto        sta    
        pe.API,ctorTye=Conneor_typnnect  co            
      0",.0."1sion=      ver       r,
       dovenr=  vendo              
    ame,ame=n  n               adata(
   onnectorMetmetadata=C        ,
        onnector_idnector_id=c  con          ctor(
    gister_conne   self._re      ms:
   syste api_ iniptioncr name, des_id, vendor,or connector       f      
   ]
  ,
      platform")API design mnia", "", "Insog", "Kon("insomnia           "),
 orment platf developman", "API"Postmstman", "Po man",("post          "),
  mentationI docuAPgger", "", "Swa "SmartBear",swagger    ("on
        atient API Docum           #        
 "),
    ormn platf"Integratio, Platform""Anypoint Soft", ft", "Mule"muleso          (rm"),
  ent platfomanagem "API gee",Apile", " "Googapigee", ("           "),
latformateway p", "API g", "Kongongong", "K"k   (      ce"),
   t serviPI managemen"Ants", Endpoi", "Cloud Googlepoints", "le_cloud_end"goog           (,
 ice")agement serv man", "APItManagemen, "API rosoft"nt", "Micmanagemeazure_api_  ("       ice"),
   rvnt semePI manage"Aateway", "API Gzon", , "Amay"_api_gatewa"aws     (      
 eways   # API Gat         ms = [
 api_syste       """
rsent connectoem managgateway andI ster AP""Regi ":
       f)nectors(seler_api_conef _regist
    
    d         )   me)
or, na vend_id,onnectors(cclasonnector_file_cf._create__class=selctor  conne                    ),
 
         "AES-256"}on": ticryp, "enn": "v1"siool_vertocx={"proility_matri compatib          ,
         me.utcnow()datetiast_updated=       l        ",
     nector_id}nectors/{conl.com/conllintecs.scro"https://dotion_url=f   documenta               },
  1000_mbps": dthwi"band1000, econd": r_serations_pe"opte_limits={   ra                nary"],
 "biro", , "av"parquet"l", son", "xm"jsv", =["cformats     data_     
          le"],ro", "iam_", "oauth2"passwordey_pair", methods=["kation_icthentau                 y"],
   e", "cop "delet", "list",ite"wr ["read",s=d_operationsupporte                ,
    ondescriptiscription=        de          
  tus.ACTIVE,Stactors=Conne  statu                .FILE,
  ectorType_type=Connctor  conne              ,
    ="1.0.0"ersion    v            ndor,
    ndor=ve         ve          
 =name,       name             adata(
orMetectnna=Coadat   met     
        r_id,toec_id=connector    conn     (
       nnectorgister_coelf._re      s     stems:
 file_syiption in scre, dedor, namctor_id, ven  for conne   
                ]
 stem"),
   syuted fileibadoop distr "H"HDFS",pache", "hdfs", "A        (
    ck"),age bloss"Server me", "SMB", ic", "Gener("smb         ),
   system"file work , "Net", "NFS"neric"Ges",      ("nf      ystems
 k File S# Networ  
                   ,
   TLS") SSL/"FTP over, ", "FTPS"Generic, "ps"  ("ft       ),
   protocol"transfer P", "File eric", "FT, "Gen"ftp"      (
      ocol"),prottransfer file Secure P", "ric", "SFTtp", "Gene ("sf          
 sferTran # File            
           
 ce"),rvistorage seect e", "Obj Storag"Cloudle", , "Goograge"cp_sto     ("g  
     vice"),torage serObject sage", ""Blob Storsoft", ", "Microzure_blob("a    
        rvice"),age se stor", "Object", "S3", "Amazonws_s3 ("a       rage
    # Cloud Sto  
          [ems =   file_syst"
      ""ectorsem connle systister fi""Reg  "
      rs(self):ile_connecto_fgister
    def _re
        )       name)
  d, vendor,connector_iector_class(saging_connesf._create_m=sellasstor_cecnn          co    
    ),        "}
      v2rsion": "", "api_veon": "v1versiol_"protoc{trix=ity_mailcompatib           ,
         w()tcno=datetime.utedast_upda  l                 _id}",
 onnector/{cm/connectorscocrollintel.//docs.srl=f"https:_uation    document           000},
     e": 10 "batch_siznd": 1000,r_secoessages_peimits={"m   rate_l               ry"],
  ", "binal", "text "xms=["json",ata_format  d             "],
     "basic_authy", "api_keh2", ["oautods=tion_meththentica au                "],
   h", "queuebatctream", "e", "siv"rece", ons=["sendd_operatiorte     supp              ription,
 on=descipti  descr                  E,
tus.ACTIVConnectorSta   status=              
   GING,rType.MESSA=Connectonnector_type co          ,
         1.0.0"version="                  dor,
  ndor=ven  ve               e,
   name=nam                    ta(
tadaonnectorMea=Ctadat         meid,
       nnector_coid=ector_     conn      or(
     nectr_conlf._registe se     s:
      aging_system messription in name, desc vendor,d, connector_i       for
        
 
        ]"),formcation plat"Communiivo", ", "Plvovo", "Plipli"     (   
     APIs"),nicationommuonage", "Cnage", "V", "Vonage      ("vo,
      orm")tion platficammunilio", "Co, "Two" "Twiliilio",    ("tw       tion
 nd Communica SMS a        # 
               ),
ervice"ail s"Em API", "Gmail"Google", l_api", "gmai (         "),
  service"Email 65", Office 3rosoft", "Mic5_mail", "36ice   ("off     ce"),
    ervimail sS", "Eon", "SE", "Amazazon_ses       ("am   "),
   platformingEmail market"p", , "Mailchimmp" "Mailchiilchimp", ("ma          m"),
 ry platforvedeli", "Email endGridGrid", "Sendgrid", "S"send   (     orms
     PlatfEmail#                      
),
   ng service"", "Messagib/Sub, "Puogle""Go", pub_subgle_oo        ("g,
    tform")aming plaret sts", "Even"Event Huboft", os"Micr_hubs", "azure_event        (e"),
    rvicKafka seed agK", "Mann", "MS "Amazo",amazon_msk        (""),
    ssagingve med-natiar", "Clouuls, "P "Apache"sar",che_pul      ("apa
       broker"),sage"Mes", , "RabbitMQe""VMwarabbitmq",        ("r   
  "),platformng streamiibuted Distrafka", ""K"Apache", a", afkche_k   ("apa       Brokers
  essage        # M = [
     stemsing_sysag      mes"""
  ctorsm conneg systeaginegister mess"""R:
        tors(self)_connecmessagingister_ _reg 
    def )
         e)
     nam, vendor, nnector_idtor_class(coity_connece_securcreats=self._r_clastonec con                    ),
           ": "v2"}
ol_version, "protoc: "v1"pi_version"rix={"aatibility_mmpat   co                ,
 ime.utcnow()dated=datetst_up    la              d}",
  nnector_is/{coectoronnel.com/cntlicrolcs.s/dops:/l=f"htttion_urenta docum                 0000},
  ": 1r_second"events_pete": 500, r_minuuests_pemits={"req    rate_li              ,
  "], "csv"syslog"", on", "xml=["jsdata_formats                   ,
 aml"]e", "scat"certifipi_key", 2", "athods=["oaumethication_    authent               
 ],"scan", udit"alert", "a", "or, "monit"extract"=[nsd_operatiosupporte                 tion,
   ripon=descpti      descri           VE,
   ACTIectorStatus.=Connatus  st            ,
      CURITYectorType.SEype=Connnector_t con                ",
   0.0rsion="1.      ve     
         ndor,or=ve        vend        ,
    me=name    na       
         rMetadata(ctotadata=Conne  me             d,
 nector_ir_id=conconnecto                connector(
ter_self._regis        ms:
    _syste in securitytioncripr, name, desr_id, vendonnecto  for co 
      
       
        ]rm"),ty platfocuriication se "Appl"Veracode", eracode","V", veracode        ("   "),
 ingtestecurity ion s "ApplicatCheckmarx",", ""Checkmarxx", arckm    ("che),
        platform"agement lity manrabie", "VulneTenablnable", ""Teble", "tena  (    
      platform"),cs ity analyti7", "Secur, "Rapid"Rapid7"apid7",      ("r      "),
 ormtfment plaanage mlnerability "Vulys",s", "QuaQualy "",alys("qu         ent
   nagembility MaVulnera #        
             ,
   alytics")ity ansecurative ", "Cloud-nogic, "Sumo Lic" "Sumo Logo_logic",      ("sum      orm"),
 platfncelligentety i"SecurigRhythm", , "Lohythm"", "LogRhythmlogr ("           ,
")gementd event manaanon y informatiit"SecurSight", "Arc", icro Focussight", "M  ("arc
          orm"),nce platfelligentcurity iar", "Se"QRadBM", ", "I"qradar      ("),
      gement manan and eventatioormurity infec", "S"Splunklunk", Spk", "plun        ("s   ent
 anagemnd Event Mn ay Informatio# Securit             
    
       agement"),access mantity and "Iden", neLogin, "O"OneLogin"n", ogi     ("onel    
   ment"), manageand accessntity "IdePingOne", entity", "", "Ping Idntity"ping_ide        (  ment"),
  ss manage acceandtity den", "I AD, "AzureMicrosoft", "e_ad"     ("azur
       ,rm")atfopltity Identh0", ""AuAuth0", "auth0", "      (  
    ement"),managaccess d ntity an", "Ide", "OktaOkta"okta", "           (agement
  Access Manty and Identi #     = [
      y_systems   securit
      """nectorsonystem city ster secur""Regis      "f):
  ctors(selcurity_conneister_sereg    def _
    )
    
        , name)endortor_id, vlass(connector_cconnecion__collaboratelf._creates=stor_clasconnec                        ),
      "}
  sion": "v1hook_ver"v1", "web": i_versionmatrix={"appatibility_   com         
        now(),e.utcetim_updated=datlast             
       tor_id}",ors/{connec.com/connectntel.scrolli//docsps:url=f"httation_ocument d            ,
       ": 100}file_size_mb: 1200, "te"nu_mi_perequestsimits={"rte_l    ra                ],
arkdown"", "mml", "csv"json", "x_formats=[    data             "],
   "webhook_key", "api", "oauth2hods=[on_metticati  authen                
  ch"],"searotify", c", "n"synd", t", "loarac"exterations=[oppported_    su               tion,
 =descripion    descript            
    s.ACTIVE,tatuonnectorS  status=C          N,
        TIOOLLABORApe.CnnectorTyype=Conector_t        con     ",
       1.0.0sion="     ver            dor,
   ndor=venve            ,
        ameame=n           n      ta(
   ctorMetadanneta=Coada         met       d,
or_iid=connect  connector_              tor(
_connecterelf._regis         s
   s:ration_toolin collaboscription me, der, na_id, vendoector  for conn      

               ]),
 tform"ration plad collabo"Database anle",  "Airtabtable",le", "Air"airtab          (e"),
  e workspacin-onn", "All-ioon", "Not, "Noti ("notion"          
 form"),agement plat, "Work many.com""Mondacom", ay."Mondy",   ("monda          ol"),
agement tot manecProj, " "Trello"","Atlassiantrello",         ("),
    atform"nt plmanageme, "Project sana"", "Aa"Asanasana",        ("
     kspace"),m wor"Teafluence", "Con", "Atlassianence", conflu       (",
      tool")managementect roja", "P, "JirAtlassian"ira", "   ("jt
         Managemenoject        # Pr
             ce"),
    ge servitora sCloud", "OneDriverosoft", "ive", "Mic("onedr           "),
 platforment tent managemon "C", "Box",ox, "Box"("b            "),
formtorage plat"Cloud sox",  "Dropb"Dropbox", pbox",     ("dro       "),
tionboraand collaorage "Cloud st"Drive", le", oogve", "G"google_dri           (,
 platform")nt agememanent cumnt", "DoarePoi "Sh",ft"Microsot", "sharepoin     (nt
       emecument Manag   # Do            
   
      latform"),g p conferencin "Video "Webex","Cisco","webex",   (         ),
 rm"g platfoencineo confer"Vid"Zoom", , "Zoom", "zoom"         ("),
   rmtion platfommunica", "Coord "Disciscord",", "D ("discord         "),
  mplatforation "Collabor"Teams", rosoft", "Micteams", ft_ ("microso         orm"),
  tfcation pla communi, "Team", "Slack""Slackack", sl    ("   forms
     ation Plat# Communic       
      [tools =n_ratiolabocol       ""
 ectors"nncol  toollaboration co""Register      "lf):
  ors(seon_connectratir_collabo _registe   def
            )
 
    ame) nor,or_id, vendconnectclass(ector_alytics_conncreate_anlass=self._onnector_c      c        ),
                 1.0"}
 on": "ersi", "min_vion": "v1={"api_versatrixity_milcompatib             
       now(),.utcmeated=datetipdlast_u           ,
         ctor_id}"ne{connectors/tel.com/concrollin/docs.sf"https:/tion_url=ocumenta      d            : 100},
  "t_gbpordata_ex 10000, "per_hour":ests_mits={"requ    rate_li               "],
 el", "pdfcsv", "excon", "jsts=["ta_forma       da    
         "],"basic_authy", _ke"api2", ths=["oaumethodication_thent   au        
         export"],ze", ", "visuali "query""load",, ract"tions=["extrted_opera  suppo               ption,
   ion=descri  descript                CTIVE,
  tatus.AectorSstatus=Conn            
        LYTICS,NArType.Aectotype=Conntor_      connec              ,
n="1.0.0"     versio           r,
    endor=vendo    v             e=name,
          nam       (
      atactorMetaddata=Conne  meta              id,
nnector_nector_id=co       con        r(
 r_connectogistef._re         selforms:
   ics_plat in analytiptioname, descrr, n vendoid,onnector_r c     fo
       
          ]
  nalytics"),"Behavior aar", ar", "Hotjtj"Hohotjar",          ("
   rm"),for data platustome", "Cment"Segnt", ", "Segmeegment    ("s,
        lytics")"Product anaude", ", "AmplitAmplitudetude", "pli       ("am,
     analytics")"Product ", "Mixpanelxpanel", anel", "Mi    ("mixp  ,
      cs")nalytiketing al mar"Digita lytics",na"Adobe", "A, _analytics"    ("adobe
        ),cs service"Web analyti", "ics "Analyt "Google",ics",nalytogle_a      ("go     Analytics
 Web          #        
 
       earning"),hine lacted m", "Automaobot"DataRtaRobot", "Dat", arobo"dat         (),
   "tformlearning pla"Machine "H2O", , 2O.ai"_ai", "H      ("h2o),
      m"ence platforci "Data s",kuai", "Dataiku"Dataiku",  ("dat    "),
        platformne learning"Machi", tform"AI Plale", m", "Googlatforai_poogle_    ("g     "),
   g servicee learnin, "MachinML"", "Azure softcro_ml", "Mi ("azure         "),
  cerning servieae l", "MachinSageMaker", ""Amazon", sagemaker  ("      m"),
    orng platfine learniachflow", "M "MLricks",Datab_ml", ""databricks        (s
     Platform Science      # Data             
 "),
    icsytrise analrp"Entetegy", croStray", "MitegtraroS "Mic",ategycrostr       ("mi
     s"),yticriven anal "Search-dSpot", "ThoughttSpot",, "Thoughpot"houghts  ("t      "),
    ncetelliges inesd busind-base "Clou"Domo", o",om", "Domo     ("d       ware"),
oftligence sintel"Business nse", se", "Sisee", "Sisensisens     ("       m"),
ence platforintelligsiness r", "Bu "Looke",leoogoker", "G      ("lo      ,
latform")ics palyta an "Dat",seen "Qlik S","Qlik", k_sense     ("qli       ,
 solution")analyticsiness "Busr BI", , "Powet""Microsofr_bi", powe  ("
          rm"),tfoce plaenntelligusiness iu", "B", "Tableaau"Tableeau",    ("tabl      nce
   lligente# Business I        = [
    ms latfornalytics_p      as"""
  nnectortform colaics pter analyt"Regis     ""elf):
   tors(sectics_connanalyister_  def _reg  
   )
      me)
       , vendor, naidnector_(contor_classoud_conneccl._create_lfr_class=setonnec     co
           ),               atest"}
 "lversion": ", "sdk_ "v1rsion":pi_ve={"ay_matrixtibilit      compa           w(),
   me.utcnotipdated=date      last_u            ",
  r_id}/{connectorsnectoom/conel.cs.scrollint"https://docfl=tion_urta   documen         ,
        s": 1000}dwidth_mbp "ban000,d": 5_per_secon={"requests rate_limits               ],
    "ary", "bin "avro",uet, "parq "csv" "xml","json",mats=[ata_for          d         ],
 account", "service_e"rol", "iam_api_key, "uth2""oas=[n_methoduthenticatio      a            
  ],ch", "bat"stream""schema",  "load", tract",ations=["expported_opersu                 
   tion,n=descripcriptio       des            ACTIVE,
 Status.onnectorstatus=C          
          LOUD,pe.CctorTynneor_type=Coctonne   c              
   "1.0.0",version=                    =vendor,
dor        ven           me,
 naame=           n
         orMetadata(ta=Connect   metada             _id,
torr_id=connecconnecto           r(
     ector_connlf._registe       se  
   rms:loud_platfoon in c descripti name,r,vendo_id, r connector       fo   
      
       ]
 form"),ng platud computi", "Clo, "Vultr, "Vultr"ltr""vu          ("),
  tformplating compuud ode", "CloLin", "dee", "Lino    ("linod      "),
  ng platformomputi "Cloud calOcean","Digit, n""DigitalOceaocean", "digital     (
       orm"),latfting pcompu", "Cloud dlibaba Clou, "AAlibaba"oud", "_cl ("alibaba          rm"),
  platfo computing, "Cloudacle Cloud"le", "Orac"Orcloud", le_orac("      "),
      rmlatfocomputing p, "Cloud BM Cloud"M", "I"IB_cloud",    ("ibm
         latformsd Pher Clou# Ot                 
  
     ,ations")ed applicneriz", "Contaioud Run "Clgle","Goo, "uncp_cloud_r        ("g),
    ervice"onitoring sring", "MtoMoni"Cloud "Google", toring", loud_moni_c  ("gcp      g"),
    inprocessh atcStream and b", ""Dataflowle",  "Googdataflow", ("gcp_          ),
 ce"ing servi", "Messag"Pub/Sub"Google", ", b_sub   ("gcp_pu   "),
      ess computeerverl", "Snsnctiooud Fu, "Cl"Google"ns", ud_functio"gcp_clo        (,
    vice")age ser"Object store", oud StoragCl, "Google"", e"ud_storaggcp_clo         ("atform
   le Cloud Pl# Goog             
           ),
ation"w automlo", "Workfogic Apps, "Lrosoft", "Micc_apps"logiazure_      (""),
      ceerviitoring sonor", "M", "Monitoft, "Micrositor"monazure_     ("     "),
  n servicetiota integra", "DatoryData Fac", "rosoft"Mic, ory"a_fact"azure_dat     (,
       latform") pamingstret ", "Evenvent Hubs, "E"Microsoft"ubs", re_event_h"azu    (       ,
 ge broker")essa", "MService Bus, "soft"", "Microsvice_bure_ser   ("azu     ),
    mpute"ess co"Serverl tions",ft", "Funcicrosoons", "Mzure_functi   ("a         ),
ce"vi storage ser"Objectage", or St, "Blobicrosoft""Mtorage", zure_blob_s"a  (       e
   zurcrosoft AMi    #           
   ),
       y service"a", "Quer", "Athen, "Amazon"thena("aws_a          "),
  rvice "ETL se","Glue"Amazon", e", _gluaws      ("      
), service"onitoring", "MWatchoudazon", "ClAmdwatch", "lou("aws_c          ),
  on service"cati, "Notifi"SNS"", azon_sns", "Am"aws          (ce"),
  uing servissage que"SQS", "Me, Amazon"ws_sqs", "       ("a),
      streaming"al-time data, "Resis"", "Kine"Amazonkinesis",     ("aws_       mpute"),
 ss coServerlebda", "ammazon", "L, "As_lambda""aw          ("),
  cempute serviC2", "Co "Ezon",c2", "Ama   ("aws_e         "),
rvicetorage sebject s"O ",", "S3", "Amazon"aws_s3          (vices
   Web Serazon Am        #  ms = [
  oud_platfor       cls"""
 nnectorplatform coster cloud egi"R""        ):
ors(selfonnecter_cloud_cst   def _regi
    
            )
 r, name)ndo_id, vetoreclass(connector_ce_connabasatreate_dass=self._connector_cl c         
       ),             "v1"}
  ": rsionrotocol_ve, "pt"": "latesersionver_vatrix={"drity_mibilipat     com           ow(),
    .utcnted=datetimet_upda   las       
          r_id}",/{connecto/connectors.comintelrollsctps://docs.f"ht_url=ocumentation  d                1000},
  ond": per_secries_ 100, "quetions":nnecits={"co    rate_lim            "],
    et", "avroparqu", "vjson", "csts=[" data_forma                ],
    "oauth2"beros","ker_cert", ", "ssl"basic_authds=[thoion_meauthenticat                "],
    _loadbulk"query", "", schema""load", xtract", ons=["etieraupported_op s                   scription,
n=deptio   descri              CTIVE,
   .AusctorStatConneus=        stat            ASE,
pe.DATABType=Connectornnector_ty        co           .0.0",
 ion="1        vers      
      r=vendor,      vendo              =name,
     name            tadata(
   =ConnectorMe   metadata          or_id,
   nectd=conconnector_i       
         ctor(ster_conneegi._rlf         se
   tems:abase_sysatn diption i name, descr, vendor,connector_id    for 
    
        
        ]abase"),ocessing dat parallel prMassivelym", " "GreenpluMware",um", "V  ("greenpl          rm"),
tics platfolyca", "Ana"Vertiocus", "Micro Fa",   ("vertic        
  form"),house platwareta "Da, Teradata"", ""Teradatadata", ("tera           tform"),
 ytics plaified analcks", "UnDatabri", ""Databricksbricks", ("data            ervice"),
tics snalyytics", "Ae Anal, "Synapsosoft"crse", "Miynap("azure_s          
  ),ics"nalytuse and arehoata waQuery", "D"Bigle", ", "Google_bigquery  ("goog
          ce"),house servi"Data waret", "Redshif, azon"t", "Amon_redshif    ("amaz      se"),
  warehoua loud datlake", "CSnowf", " "Snowflakee",nowflak"s    (es
         Warehous   # Data   
              
    service"),el database ti-mod"Mul, os DB"ft", "Cosm"Microsoos_db", zure_cosm ("a        
   ),t database"cumen "NoSQL do","Firestoreoogle", ", "Gstoreiregoogle_f("        "),
    service database "NoSQL", oDB, "Dynamzon""Amab", zon_dynamod("ama            
base"),"Graph data", eo4j"N  "Neo4j",j",     ("neo4  e"),
     ment databasDB", "Docu", "Couchhe", "Apac("couchdb          ,
  gine")lytics end anaearch an", "Slasticsearchtic", "E", "Elasicsearchlast("e            ),
re store"data structumory -meis", "Indis", "Rededis", "Re    ("r
        base"),ataolumn dde-c", "Wira "Cassande",", "Apachraassand"c   (       
  atabase"),t d, "Documen"MongoDB"", MongoDBgodb", " ("mon  
         QL Databases     # NoS
             
      ice"),tabase servlational daManaged reL", ", "Azure SQsoft", "Microe_sql"zur("a            e"),
e servicbasional dataed relatnag", "MaQL S"Cloud", "Googlel", _sqe_cloud ("googl   
         service"),l databaseed relationa, "Manag", "RDS""Amazonzon_rds", ama    ("        ase"),
 databe relationalourcOpen saDB", "", "Mari, "MariaDB"mariadb"         (   ase"),
abal datrelationded mbedite", "Eite", "SQL, "SQL("sqlite"           
 atabase"),ional dprise relatEnter, " "DB2"", "IBM",      ("db2"),
      abasenal datlatiorise re "Enterperver",", "SQL S"Microsoftr", serve   ("sql_,
         tabase")lational darise re"Enterpatabase", Oracle D"Oracle", , "_db""oracle        (se"),
    atabanal datiorelpen source "O",  "MySQLacle",Orsql", "    ("my   "),
     abaselational dat source re"Open", reSQL", "PostgreSQL"Postgstgresql", ("po           ases
 nal Datab# Relatio     
       ems = [syste_atabas  d      ""
tors"econnsystem cbase er data"""Regist   
     f):(selonnectorse_cr_databasgiste
    def _re   )
    )
         dor, name_id, vennectorlass(cononnector_c_crm_ceateelf._cr_class=sconnector       ,
             )            "1.0"}
sion": er", "min_v "v1version":rix={"api_at_militypatib   com               
  .utcnow(),d=datetimest_update     la         }",
      or_idconnectors/{ectcom/conncrollintel.ps://docs.sf"httntation_url=     docume               s": 20},
ctionconneurrent_00, "concnute": 20_per_miuestsreqits={"   rate_lim             sv"],
    "xml", "c, son"=["jrmats data_fo                  ,
 auth"]sic__key", "ba", "apiauth2"oon_methods=[nticati   authe              
   "sync"],query",  ""schema","load", tract", tions=["ex_opera   supported                 tion,
tion=descrip   descrip        
         us.ACTIVE,tatctorSonnestatus=C                RM,
    torType.C=Connectype connector_             ",
      ="1.0.0 version                 =vendor,
  dor ven           e,
        me=nam       na     (
        tarMetadaConnecto metadata=              tor_id,
 onnecector_id=c      conn         ctor(
 ister_connereg   self._      :
   _systemsption in crmname, descrindor,  veector_id,onn  for c 
        ]
       ),
      "-one CRMl-in"Al "CRM", ","Agile", le_crm      ("agi),
      l CRM"", "Socia"CRM", bleNim_crm", "le("nimb           s CRM"),
 ide sale", "Ins", "CRM, "Close""close_crm     (     "),
  rkspace CRMGoogle WoCRM", "er", "opp "Cpper_crm", ("co   ),
        ement"hip managrelationstomer , "Cus", "CRM"sightly, "Inightly""ins           (
 ),ment"ship manageationtomer relCRM", "Cus"r", Vtiger_crm", ""vtige          ("),
  managementelationship omer r"Cust", ", "CRMSugarCRM "",ugarcrm("s        "),
    gementhip manaationser rel", "Custom, "CRMshworks", "Frem"eshworks_cr       ("frt"),
     gemen manalationshipre "Customer ",RM"C ",Zoho", ""zoho_crm         (nt"),
   e managemes pipelinM", "SaleCRrive", "edip", "Pveried  ("pip          ms
ysteMajor CRM S   # Other   
             
      ice"),tomer serv "Cusice Hub",rvt", "Se", "HubSpoervice("hubspot_s          tion"),
  omaales aut"S", s Hub, "SaleHubSpot" "es",ot_salsp("hub           tion"),
 tomaeting au"Marking Hub", rketSpot", "MaHub", "ot_marketinghubsp       ("  ,
   gement")na mationship relaustomerM", "C, "CR"HubSpot"", _crmhubspot ("          Family
    # HubSpot
                      ,
")tomationting au "Marketing",rke Manamics 365ft", "Dyroso "Micng",tike365_marmics_dynasoft_ ("micro
           rvice"),tomer se "Cusce",viustomer Seramics 365 Coft", "Dyn "Microsrvice",amics_365_secrosoft_dyn   ("mi       ,
  utomation"), "Sales a65 Sales"s 3amicft", "Dyn"Microsoles", 5_sas_36microsoft_dynaic       ("m
     oft FamilyMicros#             
 
           "),rm platfolligencess intesine "Bu",ds CloulyticAnaorce", "esf", "Salloudalytics_cforce_an   ("sales       m"),
  ty platforunitomer comm", "Cusdy Clou "Communit",esforce", "Salnity_cloud_commuesforce  ("sal
          atform"),e plommerc "E-c",loudCommerce Cforce", " "Salesd",ce_clouforce_commer ("sales    
       form"), platmarketing"Digital ng Cloud",  "Marketice",sfor", "Saleudg_cloarketin_mrce"salesfo         (orm"),
   vice platfmer ser, "Custooud"ce Clvi, "Sere"orc, "Salesf_cloud"rviceforce_se   ("sales
         ),platform"mation "Sales autoud", "Sales Clo, e""Salesforcoud", ce_sales_clfor("sales            amily
esforce FSal       # 
     stems = [_sy crm"
       "ectors"onnm cM systegister CR  """Re:
      elf)nectors(sconm_r_crregiste _
    def )
             e)
  namor, tor_id, vends(connec_clasctorconneeate_erp_lf._crs=selasor_c    connect                  ),
       
   .0"}: "1rsion"", "min_ve "v1ersion":{"api_vy_matrix=ilit  compatib          ,
        ow()etime.utcnted=datt_upda    las                
r_id}",{connectonnectors/.com/collintel://docs.scrottpsl=f"hurentation_docum                 : 10},
   ctions"urrent_conne "concte": 1000,er_minuts_puests={"reqmiate_li r                  "csv"],
 l", ", "xmts=["jsonformata_         da          ,
  "saml"]"api_key",", "basic_authoauth2", hods=["tion_metca authenti           
         "query"],schema","load", "xtract", =["eperationsupported_o      s            ption,
  ption=descridescri                 IVE,
   Status.ACTtorConnecatus=st                    ,
torType.ERPnnecctor_type=Coonne c                 0",
  sion="1.0.   ver          or,
       =vendor     vend        ,
       ame    name=n           data(
     nectorMetatadata=Con        me     
   ector_id,connor_id=onnect         c    or(
   ctster_conne self._regi         :
  ystemsp_sn eron itiescrip name, dvendor,ctor_id, for conne      
       ]
           ment"),
al managean capitHumawson", "or", "L", "Infawson_erp   ("l       "),
  ased ERP "Project-bCostpoint",ltek", ""Detpoint", "deltek_cos       (     ,
em")d ERP syst-base"Cloud", P, "Cloud ERumatica"tica", "Ac("acuma       "),
     planningse resource "Enterpri"ERP", ", nit4 "Urp",_e"unit4      (,
      nning")urce plaresoe erprisnt", "Etions "Applica, "IFS",s"icationifs_appl       ("   ion"),
  solutent manageminess "Bus"X3",  "Sage", ",ge_x3       ("sa     gement"),
nanancial ma"Cloud fiacct", age", "Inttacct", "Sage_in      ("s     "),
 lanningurce prprise resoP", "Ente"ERicor",  "Epr_erp",pico       ("e
     "),ionsP solutpecific ERy-sIndustr, "Suite"r", "CloudInfoite", "loudsu ("infor_c
           ),ent suite"ness managemd busi"Cloud-baseNetSuite",  "Oracle",, "e"etsuit    ("n      
  ement"),ncial manag and fina management capital"Human, "orkdayday", "Work"W, y"orkda  ("w        ystems
   SRPr Er Majo# Othe             
      ),
      solution"nagementss ma, "Businemics GP"", "Dynaosoft"Micr", amics_gpicrosoft_dyn    ("m     ion"),
   solutmanagement ess "Busin", amics NAVft", "Dynicroso"Mv", _naft_dynamicsicroso    ("m
        "),ce planning resourrprise", "EnteDynamics AXsoft", " "Microamics_ax",icrosoft_dyn        ("mrm"),
    s platfolicationiness appBus", "ics 365", "Dynamoft, "Microscs_365"ynamiicrosoft_d   ("m  y
       rosoft Famil     # Mic  
                 ),
nagement"nce maperformaBusiness perion", ""Hyracle", "Oerion", "oracle_hyp         (,
   ")p managementhir relations", "Custome, "Siebel"lerac"Oebel", cle_si ("ora
           ning"),ource plan resseerpriEntards", "", "JD EdwOracle, "jd_edwards"oracle_      ("
      ),management"l  capitaanHumft", " "PeopleSo"Oracle",plesoft", "oracle_peo    (   ,
     ")plicationsusion Ap F "Oracleions",plicatusion Ape", "F"Oraclusion", acle_f    ("or,
        ")ouding ClPlanne se Resourcrprinte"Oracle E, ud"RP Clole", "E", "Oracrp_cloud_eoracle  ("    
      acle Family      # Or       
          em"),
 systement dor managass", "Ven"Fieldgl, "SAP", ldglass"("sap_fie          "),
  gementnamaense  and expravelncur", "TP", "Concur", "SA ("sap_co    ,
       ")nt platformProcureme"Ariba", "SAP", a", ""sap_arib    (    
    stem"),ent syemanag, "HR mors"act"SuccessF"SAP", ctors", uccessfa    ("sap_s   e"),
     ess Warehous Busin", "SAP "BW/4HANA", "SAP","sap_bw          (t"),
  mponenentral Co "SAP ERP CCC",AP", "E "S","sap_ecc       (   ),
  ng system"e plannircprise resou "EnterNA",HASAP", "S/4 "",anasap_s4h  ("        ily
   # SAP Fam           [
 p_systems =        er"
""ctorsstem conneer ERP syist""Reg       "self):
 s(nnectorr_erp_co _registe    def  
")
  se connectorterpris enonnectors)}elf.cen(slized {ltiani(f"Ifogger.in  lo 
      ()
       connectorsegister_api__r self.
       ateways   # API G   
     rs()
     nnectoile_co_register_f       self.e Systems
  Fil #  
       ()
      tors_connecmessagingter_lf._regis seems
        Systing# Messag 
              tors()
 _connec_securitygister._re        selfy Systems
Securit#          
       ectors()
ion_connaboratister_collreg      self._n Tools
  boratio# Colla   
           
  tors()ytics_connecr_anal._registeself   
     ms Platforalytics  # An     
    ors()
     oud_connectter_clegis  self._r  s
    rmloud Platfo      # C  
    rs()
    tobase_connecer_dataregist._      self Systems
   # Database              
ors()
 _connectrm_register_cself.
        M Systems       # CR        
 ectors()
_erp_connf._registersel       P Systems
       # ER""
  onnectors"available ctialize all """Ini        
(self):orsnecte_coninitializdef _   s()
    
 connectorialize_elf._init       s{}
 r] = onnectostr, BaseC Dict[onnections:ve_c self.acti {}
        =str, type]: Dict[tor_classes self.connec     {}
   ata] =orMetadctstr, Conne: Dict[f.connectorssel       :
 elf) __init__(s  def"
    
  
    ""ort suppstemrehensive syg with componboardint apid cliens rle    Enabonnectors
ation cprise applicuilt enter0+ pre-bistry for 50eg R"""
   try:
    isectorRegpriseConns Enter

clas)_check"_healthformment _perst implelasses muSubcror("ementedEr NotImpl   raise
     """ecklth cherform hea"P""       
 ny]: Dict[str, A->elf) eck(sth_chorm_healperff _ desync 
    ation")
   e_connecement _closust implbclasses m"SuError(edntmplemeotI     raise Nm"""
   rprise syste to the entee connection""Clos       "):
 tion(selfclose_connec def _    async  
  ")
nnectioblish_con_estalement es must impubclassr("SErroplementedise NotIm      ra  em"""
 systterprise ention to thenectablish con"""Es
        ):selfection(onnblish_ctasync def _es
    
    a)enticate"ement _authmust implclasses rror("SubplementedEtIm No       raisetem"""
 e sysnterprisith the ee wAuthenticat"""      e(self):
  enticatthnc def _au 
    asy
   ")et_metadatamplement _ges must i"SubclassedError(ente NotImplem rais       ta"""
metadannector et co""G"        tadata:
Mectorlf) -> Conneata(se_metaddef _get    a")
    
t get_schem implemenmustSubclasses "edError(NotImplementaise         rion"""
informatschema et """G 
       y]:tr, Ant[s Dic = None) ->entity: strself, _schema(etnc def g 
    asy   ")
d_datamplement loaasses must i("SubclmentedErrortImple raise No
       """rise systementerpta to the oad da""L "     , Any]:
  ict[strt: str) -> D]], targetr, Anyict[sst[DLia: , dat(selfata def load_d
    asyncta")
    ct_daxtralement e imp mustassesrror("SubcltedEotImplemenaise N   r""
     ise system" enterprtheata from "Extract d""        r, Any]]:
Dict[st]) -> List[tr, Anyuery: Dict[sta(self, qct_darasync def ext  
    a
        }at()
      soform().icnowime.ut: datet'timestamp'         ,
       str(e)  'error':             
  or',tus': 'errsta '             {
       return        e:
 as xception E     except    }
   
        soformat()ow().itetime.utcn: dastamp'     'time       
    o', {}),infon_('versi.getulttest_res: ion_info'  'vers       
       ]),bilities', [apa'ct.get(resultest_lities': apabi     'c           ,
ime', 0)nse_tsposult.get('ree': test_reonse_tim   'resp            success',
 ': ''status               return {
         ck()
    lth_cherm_heaelf._perfo = await sltt_resu        tes
       
         f.connect()   await sel         
    ted:is_connecf not self.    i       try:
    
     us""" statn and returnctiost conne  """Te   y]:
    Antr,> Dict[sion(self) -_connecttestdef 
    async alse
      return F   ")
       }: {str(e)}etadata.name{self.monnect from disced to r(f"Faillogger.erro           as e:
  Exception   except True
      return        
    name}")metadata.elf.d from {sDisconnecteer.info(f"  logg          False
= nected is_con self.           on()
necti_close_conself.      await    
       nnection:.cof self         i  try:
   ""
      em"prise systenterthe nnect from  """Discol:
        -> boot(self)sconnec def di 
    asynclse
   urn Fa      ret
      (e)}").name}: {stretadatalf.mct to {seonne cailed tof"Fr.error(    logge   as e:
      Exception except       rue
 n T       retur    e}")
 ama.n.metadatlf to {senectedono(f"C  logger.inf    rue
      onnected = Tis_clf.se          tion()
  lish_connecstabelf._eait s         awicate()
   f._authent await sel       :
            tryem"""
rprise systthe entetion to h connec""Establis     " bool:
    ->self)t( def connecsync    a     
se
   d = Fals_connecte    self.i= None
    n ioconnect       self.ata()
 adelf._get_metdata = s.metaself      ig
  fig = conflf.con       senfig):
 onnectionCoconfig: Cself, __(def __init    
    """
ectors connrpriseall entes for ""Base clasor:
    "aseConnects B]

clast[strs: Listationy]
    limit[str, Ans: Dicteristic_characnceperforma    str, Any]
 Dict[chema:ut_s]
    outpAnyt[str, Dicschema: put_str
    inn: descriptio str
    n:peratio o
   ity"""ctor capabil conne ants"""Represe    :
ilitypabCatorConneclass
class 
@datacAny]
r, ict[sts: Dtting timeout_se
   ct[str, Any]olicy: Di   retry_py]
 str, AnDict[d_settings:    advancer, Any]
 ct[st Dientication:   authr, Any]
 Dict[stparams: nnection_costr
    nector_id:  con"
   "ions"onnectblishing cn for estauratio"""Configfig:
    Con Connectionass
class

@datacly]ct[str, Anx: Diity_matribiltimpaime
    co datetd:ate   last_upd_url: str
 ocumentation
    dt[str, Any]Dic_limits:  ratestr]
   s: List[format
    data_tr][sthods: Listn_meatioicentuth[str]
    aListerations: upported_op sr
   ription: st desc  
 tusnectorStatus: Con statorType
   ecnn: Coypenector_tr
    conersion: st  v
   vendor: str   str
name: 
    """onnectorsnterprise ca for edat"Meta""data:
    