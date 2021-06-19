import React from 'react';
import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import CircularProgress from '@material-ui/core/CircularProgress';
import Accordion from '@material-ui/core/Accordion';
import AccordionSummary from '@material-ui/core/AccordionSummary';
import AccordionDetails from '@material-ui/core/AccordionDetails';
import Typography from '@material-ui/core/Typography';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import { BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from 'recharts';

const useStyles = makeStyles((theme) => ({
    paper: {

    },
    loading: {
        height: 250,
    },
    spinner: {
        marginBottom: 20,
    },
    root: {
        width: '100%',
    },
    heading: {
        fontSize: theme.typography.pxToRem(15),
        fontWeight: theme.typography.fontWeightRegular,
    },
    chartHeading: {
        marginBottom: 20,
    }
}));

export default function Result({ isLoading, data }) {

    const chartData = [
        {
            "name": "No attack",
            "SNR (dB)": data.TIME_DOMAIN_NO_ATTACK_SNR,
            "Robustness": data.TIME_DOMAIN_NO_ATTACK_RO,
        },
        {
            "name": "Low Pass Filter",
            "SNR (dB)": data.TIME_DOMAIN_LOW_PASS_SNR,
            "Robustness": data.TIME_DOMAIN_LOW_PASS_RO,
        },
        {
            "name": "Shearing",
            "SNR (dB)": data.TIME_DOMAIN_SHEARING_SNR,
            "Robustness": data.TIME_DOMAIN_SHEARING_RO,
        },
        {
            "name": "AWGN",
            "SNR (dB)": data.TIME_DOMAIN_AWGN_SNR,
            "Robustness": data.TIME_DOMAIN_AWGN_RO,
        },
    ]

    const classes = useStyles();

    return (
        <React.Fragment>
            <Paper className={classes.paper}>
                {isLoading ?
                    <Grid
                        container
                        direction="column"
                        justify="center"
                        alignItems="center"
                        className={classes.loading}
                    >
                        <CircularProgress className={classes.spinner} />
                        Please wait while we are processing your audio and input files. This may take a while.
                    </Grid>
                    : ""}
                {!isLoading ?
                    <Grid
                        container
                    >
                        <div className={classes.root}>
                            <Accordion defaultExpanded={true}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls="panel1a-content"
                                    id="panel1a-header"
                                >
                                    <Typography className={classes.heading}><strong>Original Image</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography>
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse malesuada lacus ex,
                                        sit amet blandit leo lobortis eget.
                                    </Typography>
                                </AccordionDetails>
                            </Accordion>
                            <Accordion defaultExpanded={true}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls="panel2a-content"
                                    id="panel2a-header"
                                >
                                    <Typography className={classes.heading}><strong>Watermarked Image</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography>
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse malesuada lacus ex,
                                        sit amet blandit leo lobortis eget.
                                    </Typography>
                                </AccordionDetails>
                            </Accordion>
                            <Accordion defaultExpanded={false}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls="panel2a-content"
                                    id="panel2a-header"
                                >
                                    <Typography className={classes.heading}><strong>Watermarked Image Under Low Pass Filter Attack</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography>
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse malesuada lacus ex,
                                        sit amet blandit leo lobortis eget.
                                    </Typography>
                                </AccordionDetails>
                            </Accordion>
                            <Accordion defaultExpanded={false}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls="panel2a-content"
                                    id="panel2a-header"
                                >
                                    <Typography className={classes.heading}><strong>Watermarked Image Under Shearing Attack</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography>
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse malesuada lacus ex,
                                        sit amet blandit leo lobortis eget.
                                    </Typography>
                                </AccordionDetails>
                            </Accordion>
                            <Accordion defaultExpanded={false}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls="panel2a-content"
                                    id="panel2a-header"
                                >
                                    <Typography className={classes.heading}><strong>Watermarked Image Under AWGN Attack</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography>
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse malesuada lacus ex,
                                        sit amet blandit leo lobortis eget.
                                    </Typography>
                                </AccordionDetails>
                            </Accordion>
                            <Accordion defaultExpanded={true}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    aria-controls="panel3a-content"
                                    id="panel3a-header"
                                >
                                    <Typography className={classes.heading}><strong>Attack Analysis</strong></Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Grid
                                        container
                                        direction="column"
                                        justify="start"
                                        alignItems="start"
                                    >
                                        <Typography className={classes.chartHeading}>SNR</Typography>
                                        <BarChart width={730} height={250} data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <Legend />
                                            <Bar dataKey="SNR (dB)" fill="#8884d8" />
                                        </BarChart>
                                        <Typography className={classes.chartHeading}>Robustness</Typography>
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
                    : ""}
            </Paper>
        </React.Fragment>
    );
}
