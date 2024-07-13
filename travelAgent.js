import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { DuckDuckGoSearch } from "@langchain/community/tools/duckduckgo_search";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { pull } from "langchain/hub";
import {
  AgentExecutor,
  createReactAgent,
  createOpenAIFunctionsAgent,
} from "langchain/agents";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HtmlToTextTransformer } from "@langchain/community/document_transformers/html_to_text";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { MozillaReadabilityTransformer } from "@langchain/community/document_transformers/mozilla_readability";

const llm = new ChatOpenAI({
  model: "gpt-3.5-turbo",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY,
});

const tools = [
  // https://js.langchain.com/v0.2/docs/integrations/tools/duckduckgo_search/
  new DuckDuckGoSearch({ maxResults: 1 }),
  // https://js.langchain.com/v0.2/docs/integrations/tools/wikipedia/
  new WikipediaQueryRun({ topKResults: 3, maxDocContentLength: 4000 }),
];

const openAIAgent = async (input, llm) => {
  const prompt = await pull("hwchase17/openai-functions-agent");
  const agent = await createOpenAIFunctionsAgent({ llm, tools, prompt });
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });
  const result = await agentExecutor.invoke({ input });

  return result.output;
};

const reActAgent = async (input, llm) => {
  const prompt = await pull("hwchase17/react");
  const agent = await createReactAgent({ llm, tools, prompt });
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });

  const result = await agentExecutor.invoke({ input });

  return result.output;
};

const loadData = async () => {
  const loader = new CheerioWebBaseLoader(
    "https://www.dicasdeviagem.com/inglaterra/",
    { selector: ".postcontentwrap" },
  );
  const docs = await loader.load();
  const splitter = RecursiveCharacterTextSplitter.fromLanguage("html");
  const transformer = new MozillaReadabilityTransformer();

  const sequence = splitter.pipe(transformer);

  const newDocuments = await sequence.invoke(docs);

  const vectorStore = await Chroma.fromDocuments(
    newDocuments,
    new OpenAIEmbeddings(),
    { collectionName: "dicas-de-viagem" },
  );

  const retriever = vectorStore.asRetriever();

  return retriever;
};

const getRelevantDocs = async (query) => {
  const retriever = await loadData();
  const relevantDocs = await retriever.invoke(query);

  return relevantDocs;
};

const supervisorAgent = async (input, llm, webContext, relevantDocs) => {
  const promptTemplate = `
    Você é um gerente de uma agência de viagens. Sua resposta final deverá ser um roteiro de viagem completo e detalhado.
    Utilize o contexto de eventos e preços de passagens, o input do usuário e também os documentos relevantes para elaborar o roteiro.
    Contexto: {webContext}
    Documentos relevantes: {relevantDocs}
    Usuário: {input}
    Assistente:`;

  const prompt = new PromptTemplate({
    inputVariables: ["webContext", "relevantDocs", "input"],
    template: promptTemplate,
  });

  const chain = RunnableSequence.from([prompt, llm]);

  const response = chain.invoke({ webContext, relevantDocs, input });

  return response;
};

const getResponse = async (input, llm) => {
  const reActOutput = await reActAgent(input, llm);
  const relevantDocs = await getRelevantDocs(input);
  const supervisorResponse = await supervisorAgent(
    input,
    llm,
    reActOutput,
    relevantDocs,
  );

  return supervisorResponse;
};

const lambaHandler = async (event) => {
  const { input } = event;
  const result = await getResponse(input, llm);

  return {
    headers: { "Content-Type": "application/json" },
    statusCode: 200,
    body: {
      message: result.content,
    },
  };
};
