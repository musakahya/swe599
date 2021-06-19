(this.webpackJsonpapp=this.webpackJsonpapp||[]).push([[0],{237:function(e,a,t){},394:function(e,a,t){"use strict";t.r(a);var n=t(0),c=t.n(n),s=t(22),r=t.n(s),i=(t(237),t(23)),o=t(454),l=t(457),d=t(462),j=t(463),h=t(458),b=t(459),O=t(453),x=t(444),m=t(455),g=t(396),p=t(440),u=t(456),f=t(452),N=t(221),S=t.n(N),v=t(203),R=t.n(v),k=t(395),_=t(445),A=t(461),W=t(464),y=t(446),C=t(70),w=t.n(C),I=t(447),T=t(451),G=t(219),B=t(220),E=t(100),P=t(97),L=t(223),F=t(3),D=Object(p.a)((function(e){return{paper:{},loading:{height:250},spinner:{marginBottom:20},root:{width:"100%"},heading:{fontSize:e.typography.pxToRem(15),fontWeight:e.typography.fontWeightRegular},chartHeading:{marginBottom:20}}}));function K(e){var a=e.isLoading,t=e.data,n=e.domain,s=[{name:"No attack","SNR (dB)":t.NO_ATTACK_SNR?t.NO_ATTACK_SNR:0,Robustness:t.NO_ATTACK_RO?t.NO_ATTACK_RO:0},{name:"Low Pass Filter","SNR (dB)":t.LOW_PASS_SNR?t.LOW_PASS_SNR:0,Robustness:t.LOW_PASS_RO?t.LOW_PASS_RO:0},{name:"Shearing","SNR (dB)":t.SHEARING_SNR?t.SHEARING_SNR:0,Robustness:t.SHEARING_RO?t.SHEARING_RO:0},{name:"AWGN","SNR (dB)":t.AWGN_SNR?t.AWGN_SNR:0,Robustness:t.AWGN_RO?t.AWGN_RO:0}],r=D();return Object(F.jsx)(c.a.Fragment,{children:Object(F.jsxs)(k.a,{className:r.paper,children:[a?Object(F.jsxs)(x.a,{container:!0,direction:"column",justify:"center",alignItems:"center",className:r.loading,children:[Object(F.jsx)(_.a,{className:r.spinner}),"Please wait while we are processing your audio and input files. This may take a while."]}):"",!a&&t&&t.AWGN_RO?Object(F.jsx)(x.a,{container:!0,children:Object(F.jsxs)("div",{className:r.root,children:[Object(F.jsxs)(A.a,{defaultExpanded:!0,children:[Object(F.jsx)(W.a,{expandIcon:Object(F.jsx)(w.a,{}),"aria-controls":"panel1a-content",id:"panel1a-header",children:Object(F.jsx)(g.a,{className:r.heading,children:Object(F.jsx)("strong",{children:"Original Image"})})}),Object(F.jsx)(y.a,{children:Object(F.jsx)("img",{src:"static/wm.bmp",alt:"Original Image"})})]}),Object(F.jsxs)(A.a,{defaultExpanded:!0,children:[Object(F.jsx)(W.a,{expandIcon:Object(F.jsx)(w.a,{}),"aria-controls":"panel2a-content",id:"panel2a-header",children:Object(F.jsx)(g.a,{className:r.heading,children:Object(F.jsx)("strong",{children:"Watermarked Image"})})}),Object(F.jsx)(y.a,{children:Object(F.jsx)("img",{src:"static/"+n+"_no_attack.png",alt:"No attack"})})]}),Object(F.jsxs)(A.a,{defaultExpanded:!0,children:[Object(F.jsx)(W.a,{expandIcon:Object(F.jsx)(w.a,{}),"aria-controls":"panel2a-content",id:"panel2a-header",children:Object(F.jsx)(g.a,{className:r.heading,children:Object(F.jsx)("strong",{children:"Watermarked Image Under Low Pass Filter Attack"})})}),Object(F.jsx)(y.a,{children:Object(F.jsx)("img",{src:"static/"+n+"_low_pass.png",alt:"Low-pass filter attack"})})]}),Object(F.jsxs)(A.a,{defaultExpanded:!0,children:[Object(F.jsx)(W.a,{expandIcon:Object(F.jsx)(w.a,{}),"aria-controls":"panel2a-content",id:"panel2a-header",children:Object(F.jsx)(g.a,{className:r.heading,children:Object(F.jsx)("strong",{children:"Watermarked Image Under Shearing Attack"})})}),Object(F.jsx)(y.a,{children:Object(F.jsx)("img",{src:"static/"+n+"_shearing.png",alt:"Shearing attack"})})]}),Object(F.jsxs)(A.a,{defaultExpanded:!0,children:[Object(F.jsx)(W.a,{expandIcon:Object(F.jsx)(w.a,{}),"aria-controls":"panel2a-content",id:"panel2a-header",children:Object(F.jsx)(g.a,{className:r.heading,children:Object(F.jsx)("strong",{children:"Watermarked Image Under AWGN Attack"})})}),Object(F.jsx)(y.a,{children:Object(F.jsx)("img",{src:"static/"+n+"_awgn.png",alt:"AWGN attack"})})]}),Object(F.jsxs)(A.a,{defaultExpanded:!0,children:[Object(F.jsx)(W.a,{expandIcon:Object(F.jsx)(w.a,{}),"aria-controls":"panel3a-content",id:"panel3a-header",children:Object(F.jsx)(g.a,{className:r.heading,children:Object(F.jsx)("strong",{children:"Attack Analysis"})})}),Object(F.jsx)(y.a,{children:Object(F.jsxs)(x.a,{container:!0,direction:"column",justify:"start",alignItems:"start",children:[Object(F.jsx)(g.a,{className:r.chartHeading,children:"SNR"}),Object(F.jsxs)(I.a,{width:730,height:250,data:s,children:[Object(F.jsx)(T.a,{strokeDasharray:"3 3"}),Object(F.jsx)(G.a,{dataKey:"name"}),Object(F.jsx)(B.a,{}),Object(F.jsx)(E.a,{}),Object(F.jsx)(P.a,{}),Object(F.jsx)(L.a,{dataKey:"SNR (dB)",fill:"#8884d8"})]}),Object(F.jsx)(g.a,{className:r.chartHeading,children:"Robustness"}),Object(F.jsxs)(I.a,{width:730,height:250,data:s,children:[Object(F.jsx)(T.a,{strokeDasharray:"3 3"}),Object(F.jsx)(G.a,{dataKey:"name"}),Object(F.jsx)(B.a,{}),Object(F.jsx)(E.a,{}),Object(F.jsx)(P.a,{}),Object(F.jsx)(L.a,{dataKey:"Robustness",fill:"#8884d8"})]})]})})]})]})}):""]})})}function H(){return Object(F.jsxs)(g.a,{variant:"body2",color:"textSecondary",align:"center",children:["Copyright \xa9 ",Object(F.jsx)(f.a,{color:"inherit",href:"https://swe-599-watermark.herokuapp.com/",children:"Audio Watermarker"})," ",(new Date).getFullYear(),"."]})}var M=Object(p.a)((function(e){return{icon:{marginRight:e.spacing(2)},heroContent:{backgroundColor:e.palette.background.paper,padding:e.spacing(8,0,6)},heroButtons:{marginTop:e.spacing(4)},cardGrid:{paddingTop:e.spacing(8),paddingBottom:e.spacing(8)},card:{height:"100%",display:"flex",flexDirection:"column"},cardMedia:{paddingTop:"56.25%"},cardContent:{flexGrow:1},footer:{backgroundColor:e.palette.background.paper,padding:e.spacing(6)},button:{display:"block",marginTop:e.spacing(2)},formControl:{margin:e.spacing(1),minWidth:120}}}));function U(){var e=M(),a=c.a.useState(),t=Object(i.a)(a,2),s=t[0],r=t[1],p=c.a.useState(),f=Object(i.a)(p,2),N=f[0],v=f[1],k=c.a.useState(!1),_=Object(i.a)(k,2),A=_[0],W=_[1],y=c.a.useState(""),C=Object(i.a)(y,2),w=C[0],I=C[1],T=c.a.useState(!1),G=Object(i.a)(T,2),B=G[0],E=G[1],P=c.a.useState(""),L=Object(i.a)(P,2),D=L[0],U=L[1],z=c.a.useState(!1),J=Object(i.a)(z,2),Y=J[0],q=J[1],Q=c.a.useState(""),V=Object(i.a)(Q,2),X=V[0],Z=V[1],$=c.a.useState(!1),ee=Object(i.a)($,2),ae=ee[0],te=ee[1];Object(n.useEffect)((function(){A&&W(!1)}),[X]);return Object(F.jsxs)(c.a.Fragment,{children:[Object(F.jsx)(O.a,{}),Object(F.jsx)(o.a,{position:"relative",children:Object(F.jsxs)(m.a,{children:[Object(F.jsx)(S.a,{fontSize:"large",className:e.icon}),Object(F.jsx)(g.a,{variant:"h6",color:"inherit",noWrap:!0,children:"Audio Watermarker"})]})}),Object(F.jsxs)("main",{children:[Object(F.jsx)("div",{className:e.heroContent,children:Object(F.jsxs)(u.a,{maxWidth:"sm",children:[Object(F.jsx)(g.a,{component:"h1",variant:"h2",align:"center",color:"textPrimary",gutterBottom:!0,children:"Protect your audio"}),Object(F.jsxs)(g.a,{variant:"h5",align:"center",color:"textSecondary",paragraph:!0,children:["Embed secret images into your audio files to claim ownership. Compare embedding in different domains such as"," ",Object(F.jsx)("strong",{children:"time, cosine,"})," and ",Object(F.jsx)("strong",{children:"wavelet"}),". Pick the one that works best."]}),Object(F.jsx)("div",{className:e.heroButtons,children:Object(F.jsxs)(x.a,{container:!0,spacing:2,justify:"center",children:[Object(F.jsx)(x.a,{item:!0,children:Object(F.jsx)(l.a,{variant:"contained",color:"primary",children:"Get Started"})}),Object(F.jsx)(x.a,{item:!0,children:Object(F.jsx)(l.a,{variant:"outlined",color:"primary",children:"Learn More"})})]})})]})}),Object(F.jsxs)(u.a,{className:e.cardGrid,maxWidth:"md",children:[Object(F.jsxs)(h.a,{className:e.formControl,fullWidth:!0,children:[Object(F.jsx)(d.a,{id:"demo-controlled-open-select-label",children:"Select Audio"}),Object(F.jsxs)(b.a,{labelId:"demo-controlled-open-select-label",id:"demo-controlled-open-select",open:B,onClose:function(){E(!1)},onOpen:function(){E(!0)},value:w,onChange:function(e){I(e.target.value)},children:[Object(F.jsx)(j.a,{value:"",children:Object(F.jsx)("em",{children:"None"})}),Object(F.jsx)(j.a,{value:"100grand",children:"100grand.wav"})]})]}),Object(F.jsxs)(h.a,{className:e.formControl,fullWidth:!0,children:[Object(F.jsx)(d.a,{id:"demo-controlled-open-select-label",children:"Select Image"}),Object(F.jsxs)(b.a,{labelId:"demo-controlled-open-select-label",id:"demo-controlled-open-select",open:Y,onClose:function(){q(!1)},onOpen:function(){q(!0)},value:D,onChange:function(e){U(e.target.value)},children:[Object(F.jsx)(j.a,{value:"",children:Object(F.jsx)("em",{children:"None"})}),Object(F.jsx)(j.a,{value:"lena",children:"lena.bmp"})]})]}),Object(F.jsxs)(h.a,{className:e.formControl,fullWidth:!0,children:[Object(F.jsx)(d.a,{id:"demo-controlled-open-select-label",children:"Select Domain"}),Object(F.jsxs)(b.a,{labelId:"demo-controlled-open-select-label",id:"demo-controlled-open-select",open:ae,onClose:function(){te(!1)},onOpen:function(){te(!0)},value:X,onChange:function(e){Z(e.target.value)},children:[Object(F.jsx)(j.a,{value:"",children:Object(F.jsx)("em",{children:"None"})}),Object(F.jsx)(j.a,{value:"time_domain",children:"Time"}),Object(F.jsx)(j.a,{value:"cosine",children:"Discrete Cosine Transform"}),Object(F.jsx)(j.a,{value:"wavelet",children:"Discrete Wavelet Transform"})]})]}),Object(F.jsx)(h.a,{className:e.formControl,fullWidth:!0,children:Object(F.jsx)(l.a,{disabled:""===X||""===D||""===w,variant:"contained",color:"primary",onClick:function(){r(!0),"time_domain"===X||"wavelet"===X||"cosine"===X?R.a.get("/"+X+"/").then((function(e){console.log(e.data),v(e.data),W(!0),r(!1)})).catch((function(e){console.log(e)})).then((function(){})):console.log(X)},children:!0===s?"Working...":A?"Completed":"Start Watermarking"})}),!0===s||!1===s?Object(F.jsx)(h.a,{className:e.formControl,fullWidth:!0,children:N&&A?Object(F.jsx)(K,{isLoading:s,data:N,domain:X}):""}):""]})]}),Object(F.jsxs)("footer",{className:e.footer,children:[Object(F.jsx)(g.a,{variant:"subtitle1",align:"center",color:"textSecondary",component:"p",children:"Protect your audio"}),Object(F.jsx)(H,{})]})]})}var z=function(e){e&&e instanceof Function&&t.e(3).then(t.bind(null,466)).then((function(a){var t=a.getCLS,n=a.getFID,c=a.getFCP,s=a.getLCP,r=a.getTTFB;t(e),n(e),c(e),s(e),r(e)}))};r.a.render(Object(F.jsx)(c.a.StrictMode,{children:Object(F.jsx)(U,{})}),document.getElementById("root")),z()}},[[394,1,2]]]);
//# sourceMappingURL=main.9dce2e9e.chunk.js.map