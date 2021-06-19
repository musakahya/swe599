import React from 'react';
import AppBar from '@material-ui/core/AppBar';
import Button from '@material-ui/core/Button';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import CssBaseline from '@material-ui/core/CssBaseline';
import Grid from '@material-ui/core/Grid';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Link from '@material-ui/core/Link';
import WbAutoIcon from '@material-ui/icons/WbAuto';
import axios from 'axios';

import Result from './Result'

function Copyright() {
  return (
    <Typography variant="body2" color="textSecondary" align="center">
      {'Copyright © '}
      <Link color="inherit" href="https://swe-599-watermark.herokuapp.com/">
        Audio Watermarker
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
  );
}

const useStyles = makeStyles((theme) => ({
  icon: {
    marginRight: theme.spacing(2),
  },
  heroContent: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(8, 0, 6),
  },
  heroButtons: {
    marginTop: theme.spacing(4),
  },
  cardGrid: {
    paddingTop: theme.spacing(8),
    paddingBottom: theme.spacing(8),
  },
  card: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
  },
  cardMedia: {
    paddingTop: '56.25%', // 16:9
  },
  cardContent: {
    flexGrow: 1,
  },
  footer: {
    backgroundColor: theme.palette.background.paper,
    padding: theme.spacing(6),
  },
  button: {
    display: 'block',
    marginTop: theme.spacing(2),
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 120,
  },
}));

export default function App() {
  const classes = useStyles();

  const [isLoading, setLoading] = React.useState();
  const [data, setData] = React.useState();

  const [audio, setAudio] = React.useState('');
  const [openAudio, setOpenAudio] = React.useState(false);

  const [image, setImage] = React.useState('');
  const [openImage, setOpenImage] = React.useState(false);

  const [domain, setDomain] = React.useState('');
  const [openDomain, setOpenDomain] = React.useState(false);

  const handleChangeAudio = (event) => {
    setAudio(event.target.value);
  };

  const handleCloseAudio = () => {
    setOpenAudio(false);
  };

  const handleOpenAudio = () => {
    setOpenAudio(true);
  };

  const handleChangeImage = (event) => {
    setImage(event.target.value);
  };

  const handleCloseImage = () => {
    setOpenImage(false);
  };

  const handleOpenImage = () => {
    setOpenImage(true);
  };

  const handleChangeDomain = (event) => {
    setDomain(event.target.value);
  };

  const handleCloseDomain = () => {
    setOpenDomain(false);
  };

  const handleOpenDomain = () => {
    setOpenDomain(true);
  };

  const handleStart = () => {
    setLoading(true);
    let urlPath = '';
    if(domain === "time") urlPath = "time_domain";
    else if(domain === "wavelet") urlPath = "wavelet";
    else if(domain === "cosine") urlPath = "cosine";
    else;
    axios.get('https://swe-599-watermark.herokuapp.com/' + urlPath + '/')
      .then(function (response) {
        // handle success
        setData(response.data);
        setLoading(false);
      })
      .catch(function (error) {
        // handle error
        console.log(error);
      })
      .then(function () {
        // always executed
      });
  };

  return (
    <React.Fragment>
      <CssBaseline />
      <AppBar position="relative">
        <Toolbar>
          <WbAutoIcon fontSize="large" className={classes.icon} />
          <Typography variant="h6" color="inherit" noWrap>
            Audio Watermarker
          </Typography>
        </Toolbar>
      </AppBar>
      <main>
        {/* Hero unit */}
        <div className={classes.heroContent}>
          <Container maxWidth="sm">
            <Typography component="h1" variant="h2" align="center" color="textPrimary" gutterBottom>
              Protect your audio
            </Typography>
            <Typography variant="h5" align="center" color="textSecondary" paragraph>
              Embed secret images into your audio files to claim ownership. Compare embedding in different domains such as <strong>time, cosine,</strong> and <strong>wavelet</strong>. Pick the one that works best.
            </Typography>
            <div className={classes.heroButtons}>
              <Grid container spacing={2} justify="center">
                <Grid item>
                  <Button variant="contained" color="primary">
                    Get Started
                  </Button>
                </Grid>
                <Grid item>
                  <Button variant="outlined" color="primary">
                    Learn More
                  </Button>
                </Grid>
              </Grid>
            </div>
          </Container>
        </div>
        <Container className={classes.cardGrid} maxWidth="md">
          {/* End hero unit */}
          <FormControl className={classes.formControl} fullWidth>
            <InputLabel id="demo-controlled-open-select-label">Select Audio</InputLabel>
            <Select
              labelId="demo-controlled-open-select-label"
              id="demo-controlled-open-select"
              open={openAudio}
              onClose={handleCloseAudio}
              onOpen={handleOpenAudio}
              value={audio}
              onChange={handleChangeAudio}
            >
              <MenuItem value="">
                <em>None</em>
              </MenuItem>
              <MenuItem value={"100grand"}>100grand.wav</MenuItem>
            </Select>
          </FormControl>
          <FormControl className={classes.formControl} fullWidth>
            <InputLabel id="demo-controlled-open-select-label">Select Image</InputLabel>
            <Select
              labelId="demo-controlled-open-select-label"
              id="demo-controlled-open-select"
              open={openImage}
              onClose={handleCloseImage}
              onOpen={handleOpenImage}
              value={image}
              onChange={handleChangeImage}
            >
              <MenuItem value="">
                <em>None</em>
              </MenuItem>
              <MenuItem value={"lena"}>lena.bmp</MenuItem>
            </Select>
          </FormControl>
          <FormControl className={classes.formControl} fullWidth>
            <InputLabel id="demo-controlled-open-select-label">Select Domain</InputLabel>
            <Select
              labelId="demo-controlled-open-select-label"
              id="demo-controlled-open-select"
              open={openDomain}
              onClose={handleCloseDomain}
              onOpen={handleOpenDomain}
              value={domain}
              onChange={(e) => handleChangeDomain(e)}
            >
              <MenuItem value="">
                <em>None</em>
              </MenuItem>
              <MenuItem value={"time"}>Time</MenuItem>
              <MenuItem value={"dct"}>Discrete Cosine Transform</MenuItem>
              <MenuItem value={"dwt"}>Discrete Wavelet Transform</MenuItem>
            </Select>
          </FormControl>
          <FormControl className={classes.formControl} fullWidth>
            <Button disabled={domain === "" || image === "" || audio === ""} variant="contained" color="primary" onClick={handleStart}>
              {isLoading === true ? "Working..." : (isLoading === false ? "Completed" : "Start Watermarking")}
            </Button>
          </FormControl>
          {
            isLoading === true || isLoading === false ?
              <FormControl className={classes.formControl} fullWidth>
                {data && data.TIME_DOMAIN_NO_ATTACK_SNR ? <Result isLoading={isLoading} data={data}/> : ""}
              </FormControl>
              : ""
          }
        </Container>
      </main>
      {/* Footer */}
      <footer className={classes.footer}>
        <Typography variant="subtitle1" align="center" color="textSecondary" component="p">
          Protect your audio
        </Typography>
        <Copyright />
      </footer>
      {/* End footer */}
    </React.Fragment>
  );
}
