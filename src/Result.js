import React from "react";
import Grid from "@material-ui/core/Grid";
import { makeStyles } from "@material-ui/core/styles";
import Paper from "@material-ui/core/Paper";
import CircularProgress from "@material-ui/core/CircularProgress";
import Accordion from "@material-ui/core/Accordion";
import AccordionSummary from "@material-ui/core/AccordionSummary";
import AccordionDetails from "@material-ui/core/AccordionDetails";
import Typography from "@material-ui/core/Typography";
import ExpandMoreIcon from "@material-ui/icons/ExpandMore";
import {
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Bar,
} from "recharts";

const useStyles = makeStyles((theme) => ({
  paper: {},
  loading: {
    height: 250,
  },
  spinner: {
    marginBottom: 20,
  },
  root: {
    width: "100%",
  },
  heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
  },
  chartHeading: {
    marginBottom: 20,
  },
}));

export default function Result({ isLoading, data, domain }) {
  const chartData = [
    {
      name: "No attack",
      "SNR (dB)": data.NO_ATTACK_SNR ? data.NO_ATTACK_SNR : 0,
      Robustness: data.NO_ATTACK_RO ? data.NO_ATTACK_RO :  0,
    },
    {
      name: "Low Pass Filter",
      "SNR (dB)": data.LOW_PASS_SNR ?  data.LOW_PASS_SNR : 0,
      Robustness: data.LOW_PASS_RO ?  data.LOW_PASS_RO : 0,
    },
    {
      name: "Shearing",
      "SNR (dB)": data.SHEARING_SNR ? data.SHEARING_SNR : 0,
      Robustness: data.SHEARING_RO ? data.SHEARING_RO : 0,
    },
    {
      name: "AWGN",
      "SNR (dB)": data.AWGN_SNR ? data.AWGN_SNR : 0,
      Robustness: data.AWGN_RO ? data.AWGN_RO : 0,
    },
  ];

  const classes = useStyles();

  return (
    <React.Fragment>
      <Paper className={classes.paper}>
        {isLoading ? (
          <Grid
            container
            direction="column"
            justify="center"
            alignItems="center"
            className={classes.loading}
          >
            <CircularProgress className={classes.spinner} />
            Please wait while we are processing your audio and input files. This
            may take a while.
          </Grid>
        ) : (
          ""
        )}
        {!isLoading && data && data.AWGN_RO ? (
          <Grid container>
            <div className={classes.root}>
              <Accordion defaultExpanded={true}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel1a-content"
                  id="panel1a-header"
                >
                  <Typography className={classes.heading}>
                    <strong>Original Image</strong>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <img src="static/wm.bmp" alt="Original Image" />
                </AccordionDetails>
              </Accordion>
              <Accordion defaultExpanded={true}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel2a-content"
                  id="panel2a-header"
                >
                  <Typography className={classes.heading}>
                    <strong>Watermarked Image</strong>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <img src={"static/" + domain + "_no_attack.png"} alt="No attack" />
                </AccordionDetails>
              </Accordion>
              <Accordion defaultExpanded={true}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel2a-content"
                  id="panel2a-header"
                >
                  <Typography className={classes.heading}>
                    <strong>
                      Watermarked Image Under Low Pass Filter Attack
                    </strong>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <img src={"static/" + domain + "_low_pass.png"} alt="Low-pass filter attack" />
                </AccordionDetails>
              </Accordion>
              <Accordion defaultExpanded={true}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel2a-content"
                  id="panel2a-header"
                >
                  <Typography className={classes.heading}>
                    <strong>Watermarked Image Under Shearing Attack</strong>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <img src={"static/" + domain + "_shearing.png"} alt="Shearing attack" />
                </AccordionDetails>
              </Accordion>
              <Accordion defaultExpanded={true}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel2a-content"
                  id="panel2a-header"
                >
                  <Typography className={classes.heading}>
                    <strong>Watermarked Image Under AWGN Attack</strong>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <img src={"static/" + domain + "_awgn.png"} alt="AWGN attack" />
                </AccordionDetails>
              </Accordion>
              <Accordion defaultExpanded={true}>
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  aria-controls="panel3a-content"
                  id="panel3a-header"
                >
                  <Typography className={classes.heading}>
                    <strong>Attack Analysis</strong>
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid
                    container
                    direction="column"
                    justify="start"
                    alignItems="start"
                  >
                    <Typography className={classes.chartHeading}>
                      SNR
                    </Typography>
                    <BarChart width={730} height={250} data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="SNR (dB)" fill="#8884d8" />
                    </BarChart>
                    <Typography className={classes.chartHeading}>
                      Robustness
                    </Typography>
                    <BarChart width={730} height={250} data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="Robustness" fill="#8884d8" />
                    </BarChart>
                  </Grid>
                </AccordionDetails>
              </Accordion>
            </div>
          </Grid>
        ) : (
          ""
        )}
      </Paper>
    </React.Fragment>
  );
}
