import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link rel="icon" href="/favicon.ico" />
        <meta name="description" content="Institutional-grade quantitative finance pipeline combining classical portfolio theory, ML forecasting, and Bayesian inference" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
