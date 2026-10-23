import parse from 'html-react-parser';
import { readPage } from '../lib/site-pages';
import PageScripts from './page-scripts';

export default function ResearchPage({ name }) {
  const page = readPage(name);
  return <>
    {page.styles.map(href => <link key={href} rel="stylesheet" href={`/${href}`} precedence="site" />)}
    {parse(page.markup)}
    <PageScripts scripts={page.scripts} bodyClass={page.bodyClass} />
  </>;
}
