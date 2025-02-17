import React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

function Message({ sender, text, sources }) {
  return (
    <div
      className={`${
        sender === "user" ? "ml-auto bg-gray-300" : ""
      } min-h-16 max-w-[75%] min-w-32 h-max rounded-lg p-2 m-2 break-words`}
    >
      <ReactMarkdown
        className={"prose"}
        components={{
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            return !inline && match ? (
              <SyntaxHighlighter
                style={oneDark}
                language={match[1]}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, "")}
              </SyntaxHighlighter>
            ) : (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
      {sender === "chatbot" ? (
        <div className="flex flex-col mt-2">
          {sources && sources.length > 0 && (
            <>
              <p className="text-sm font-semibold">------------------------------------------------</p>
              <p className="text-sm font-semibold">Sources:</p>
              {sources.map((singleSource, index) => (
                <p key={index} className="text-xs text-gray-600">
                  Source: {singleSource.pdf} - Page Number: {singleSource.page} 
                </p>
              ))}
            </>
          )}
        </div>
      ) : (
        null
      )}
    </div>
  );
}

export default Message;
