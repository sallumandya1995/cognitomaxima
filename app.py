import ai21
import os 
import gradio as gr
ai21.api_key = os.environ['API_KEY']
def articletitle(article_prompt:str):

  output=ai21.Completion.execute( 
  model="j2-large",
  prompt=f"""Article title on digital trust: Earning digital trust: Where to invest today and tomorrow
            Article title on organizational trust: Can you measure trust within your organization?
            Article title on brainstorming: Bias busters: A better way to brainstorm
            Article title on dreaming: How dreams effect the brain
            Article title on {article_prompt}:""",
  numResults=1,
  maxTokens=64,
  temperature=1,
  topKReturn=0.55,
  topP=0.55,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["↵"]
  )
  
  return output.completions[0]['data']['text']

def outlinecreator(article_title:str):

  output1=ai21.Completion.execute(
  model="j2-large",
  prompt=f"""======================================
            Write 6 sections to a great blog post for the following title and end it with a conclusion.
            Blog title: How to start a personal blog
            Blog sections:1.Define your niche and target audience\n2.Choose a blogging platform and domain name\n3.Design your blog and create content\n4.Promote your blog through social media and networking\n5.Build a community through engagement and interaction\n6.Monitor and track your blog's performance\nConclusion
            ======================================
            Write 6 sections to a great blog post for the following title and end it with a conclusion.
            Blog title: A real-world example on Improving JavaScript performance
            Blog sections:1.Introduction to JavaScript performance optimization\n2.Understanding the current performance issues\n3.Implementing best practices and coding standards\n4.Optimizing the code using profiling and debugging tools\n5.Testing and validating the performance improvements\n6.Monitoring and maintaining the optimized code\nConclusion'
            ======================================
            Write 6 sections to a great blog post for the following title and end it with a conclusion.
            Blog title: Is a Happy Life Different from a Meaningful One?
            Blog sections:1.Introduction: Defining happiness and meaning\n2.The Differences Between Happiness and Meaning\n3.The Importance of Happiness\n4.The Importance of Meaning\n5.How to Balance Happiness and Meaning in Your Life\n6.Finding your own path to a happy and meaningful life\nConclusion
            ======================================
            Write 6 sections to a great blog post for the following title and end it with a conclusion.
            Blog title:{article_title}
            Blog Sections:""",
  numResults=1,
  maxTokens=296,
  temperature=1,
  topKReturn=0.55,
  topP=0.55,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 185,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0.4,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["==="]
  )
  return output1.completions[0]['data']['text'].strip()

def paragraphgeneration(article_title,blog_title):
  output2=ai21.Completion.execute(
  model="j2-large",
  prompt=f"""
  Title: What the science says about global warming
  Blog Sections: 
  1. Global warming is a naturally occurring phenomenon
  2. Human activity has altered the natural cycles of earth
  3. Climate change is causing more extreme weather
  4. The effects of climate change are devastating
  5. What can we do about climate change?
  Write a 20 lines for each blog title
  Blog:
  1.Global warming is a naturally occurring phenomenon:
  Global warming is a phenomenon that occurs naturally, and it has been happening for millions of years. Earth's climate has gone through natural cycles of warming and cooling, but what we are experiencing today is different. The rapid increase in temperatures is due to human activities like burning fossil fuels, deforestation, and industrial processes. Scientists have studied historical records, ice cores, and other data to confirm that the current warming trend is not a natural occurrence.
  2.Human activity has altered the natural cycles of earth:
  Human activity has significantly altered the natural cycles of the earth, leading to climate change. The burning of fossil fuels has increased the concentration of greenhouse gases in the atmosphere, trapping more heat and causing global temperatures to rise. Deforestation has also contributed to climate change by reducing the number of trees that absorb carbon dioxide. Industrial processes, agriculture, and transportation also contribute to greenhouse gas emissions. It's important to acknowledge the role humans have played in altering the natural cycles of the earth and take responsibility for our actions.
  3.Climate change is causing more extreme weather:
  Climate change is causing more extreme weather events like heatwaves, droughts, hurricanes, and floods. The increased concentration of greenhouse gases in the atmosphere traps more heat, leading to warmer temperatures and more frequent heatwaves. Droughts are becoming more severe in some regions, while other areas experience increased rainfall and flooding. Hurricanes are becoming more powerful due to warmer ocean temperatures. It's essential to take action to reduce greenhouse gas emissions to mitigate the impact of extreme weather events.
  4.The effects of climate change are devastating:
  The effects of climate change are devastating and have far-reaching consequences. Rising sea levels are causing coastal flooding, which can displace people and destroy infrastructure. Changes in rainfall patterns can affect agriculture, leading to food shortages and increased prices. The loss of biodiversity due to habitat destruction and other factors can have ripple effects throughout ecosystems. Climate change also exacerbates health problems, like asthma and heatstroke. It's essential to take action to prevent the worst effects of climate change.
  5.What can we do about climate change?
  We can take action to reduce greenhouse gas emissions and mitigate the impact of climate change. Individuals can reduce their carbon footprint by using public transportation, eating less meat, and using energy-efficient appliances. Governments can implement policies like carbon taxes and invest in renewable energy sources like solar and wind power. Businesses can also take steps to reduce their carbon footprint and adopt sustainable practices. It's essential to work together to address the problem of climate change and prevent the worst impacts on our planet.
  ============================================================================================================================================================================================================================================================================================================
  Title: The art and science of storytelling
  Blog Sections: 
  1. Why Storytelling is important
  2. What makes a good story?
  3. What tools do I need for storytelling?
  4. How to create stories
  5. How to tell stories
  6. Storytelling tips
  Write a 20 lines for each blog title
  Blog:
  1.Why Storytelling is important:
  Storytelling is an important form of human communication that has been around for thousands of years. Stories are a way to connect with people, share ideas, and convey emotions. They can inspire, entertain, educate, and motivate. Stories are also a way to preserve culture and history, passing down important lessons and traditions from generation to generation.
  2.What makes a good story?
  A good story has several key elements that make it compelling and memorable. It should have a clear beginning, middle, and end, with well-defined characters and a plot that keeps the audience engaged. The story should evoke emotions and have a central theme or message. A good story also has a unique perspective or twist that sets it apart from others.
  3.What tools do I need for storytelling?
  To tell a good story, you need several tools, including a strong understanding of the story's structure and elements, good writing skills, and the ability to use tone, voice, and pacing effectively. Visual aids like images, videos, or props can also enhance storytelling. Good public speaking skills and an understanding of your audience are also important.
  4.How to create stories:
  Creating a story can be a fun and rewarding process. Start by brainstorming ideas and developing the characters and plot. Create a strong opening that captures the audience's attention and develops the story through conflict and resolution. Focus on the central theme or message of the story and consider the emotions you want to convey. Revisions and editing are crucial to creating a polished and effective story.
  5. How to tell stories:
  Telling a story involves more than just reciting the words. It's important to use body language, voice inflection, and pacing to engage the audience and create a memorable experience. Start with a strong opening and hook the audience with a compelling story arc. Use clear and concise language, and be mindful of your audience's attention span. Practice, rehearsal, and feedback from others can help improve storytelling skills.
  6.Storytelling tips:
  Some storytelling tips include: know your audience, keep the story concise and focused, use humor when appropriate, incorporate sensory details, and use visual aids. Practice storytelling in front of others to get feedback and improve your skills. Consider the context and purpose of the story, and tailor it accordingly. Finally, be authentic and passionate about the story you are telling, and let your personality shine through.
  ============================================================================================================================================================================================================================================================================================================
  Title: {article_title}
  Blog Sections:
  {blog_title}
  Write a 20 lines for each blog title
  Blog:
  
  """
,  
   numResults=1,
  maxTokens=800,
  temperature=1,
  topKReturn=0.7,
  topP=0.7,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["===="]
  )
  print(f"""Title: What the science says about global warming
  Blog Sections: 
  1. Global warming is a naturally occurring phenomenon
  2. Human activity has altered the natural cycles of earth
  3. Climate change is causing more extreme weather
  4. The effects of climate change are devastating
  5. What can we do about climate change?
  Write a 20 lines for each blog title
  Blog:
  1.Global warming is a naturally occurring phenomenon:
  Global warming is a phenomenon that occurs naturally, and it has been happening for millions of years. Earth's climate has gone through natural cycles of warming and cooling, but what we are experiencing today is different. The rapid increase in temperatures is due to human activities like burning fossil fuels, deforestation, and industrial processes. Scientists have studied historical records, ice cores, and other data to confirm that the current warming trend is not a natural occurrence.
  2.Human activity has altered the natural cycles of earth:
  Human activity has significantly altered the natural cycles of the earth, leading to climate change. The burning of fossil fuels has increased the concentration of greenhouse gases in the atmosphere, trapping more heat and causing global temperatures to rise. Deforestation has also contributed to climate change by reducing the number of trees that absorb carbon dioxide. Industrial processes, agriculture, and transportation also contribute to greenhouse gas emissions. It's important to acknowledge the role humans have played in altering the natural cycles of the earth and take responsibility for our actions.
  3.Climate change is causing more extreme weather:
  Climate change is causing more extreme weather events like heatwaves, droughts, hurricanes, and floods. The increased concentration of greenhouse gases in the atmosphere traps more heat, leading to warmer temperatures and more frequent heatwaves. Droughts are becoming more severe in some regions, while other areas experience increased rainfall and flooding. Hurricanes are becoming more powerful due to warmer ocean temperatures. It's essential to take action to reduce greenhouse gas emissions to mitigate the impact of extreme weather events.
  4.The effects of climate change are devastating:
  The effects of climate change are devastating and have far-reaching consequences. Rising sea levels are causing coastal flooding, which can displace people and destroy infrastructure. Changes in rainfall patterns can affect agriculture, leading to food shortages and increased prices. The loss of biodiversity due to habitat destruction and other factors can have ripple effects throughout ecosystems. Climate change also exacerbates health problems, like asthma and heatstroke. It's essential to take action to prevent the worst effects of climate change.
  5.What can we do about climate change?
  We can take action to reduce greenhouse gas emissions and mitigate the impact of climate change. Individuals can reduce their carbon footprint by using public transportation, eating less meat, and using energy-efficient appliances. Governments can implement policies like carbon taxes and invest in renewable energy sources like solar and wind power. Businesses can also take steps to reduce their carbon footprint and adopt sustainable practices. It's essential to work together to address the problem of climate change and prevent the worst impacts on our planet.
  ============================================================================================================================================================================================================================================================================================================
  Title: The art and science of storytelling
  Blog Sections: 
  1. Why Storytelling is important
  2. What makes a good story?
  3. What tools do I need for storytelling?
  4. How to create stories
  5. How to tell stories
  6. Storytelling tips
  Write a 20 lines for each blog title
  Blog:
  1.Why Storytelling is important:
  Storytelling is an important form of human communication that has been around for thousands of years. Stories are a way to connect with people, share ideas, and convey emotions. They can inspire, entertain, educate, and motivate. Stories are also a way to preserve culture and history, passing down important lessons and traditions from generation to generation.
  2.What makes a good story?
  A good story has several key elements that make it compelling and memorable. It should have a clear beginning, middle, and end, with well-defined characters and a plot that keeps the audience engaged. The story should evoke emotions and have a central theme or message. A good story also has a unique perspective or twist that sets it apart from others.
  3.What tools do I need for storytelling?
  To tell a good story, you need several tools, including a strong understanding of the story's structure and elements, good writing skills, and the ability to use tone, voice, and pacing effectively. Visual aids like images, videos, or props can also enhance storytelling. Good public speaking skills and an understanding of your audience are also important.
  4.How to create stories:
  Creating a story can be a fun and rewarding process. Start by brainstorming ideas and developing the characters and plot. Create a strong opening that captures the audience's attention and develops the story through conflict and resolution. Focus on the central theme or message of the story and consider the emotions you want to convey. Revisions and editing are crucial to creating a polished and effective story.
  5. How to tell stories:
  Telling a story involves more than just reciting the words. It's important to use body language, voice inflection, and pacing to engage the audience and create a memorable experience. Start with a strong opening and hook the audience with a compelling story arc. Use clear and concise language, and be mindful of your audience's attention span. Practice, rehearsal, and feedback from others can help improve storytelling skills.
  6/Storytelling tips:
  Some storytelling tips include: know your audience, keep the story concise and focused, use humor when appropriate, incorporate sensory details, and use visual aids. Practice storytelling in front of others to get feedback and improve your skills. Consider the context and purpose of the story, and tailor it accordingly. Finally, be authentic and passionate about the story you are telling, and let your personality shine through.
  ============================================================================================================================================================================================================================================================================================================
  Title: {article_title}
  Blog Sections:
  {blog_title}
  
  """)
  return output2.completions[0]['data']['text']

# with gr.Blocks() as demo: 
#   with gr.Row():
#     with gr.Column():
#       input_topic = gr.Textbox(label="Topic (Be descriptive about it and press enter to generate a title)",placeholder="What the science says about global warming? (Press enter to continue)")
#       input_title = gr.Button("Generate Blog Sections")
      
      
#     with gr.Column():
#       output_title = gr.Textbox(label="Generated Title")
#       output_blog_sections = gr.Textbox(label="Blog Sections")
#       complete_blog_text = gr.Textbox(label="Blog")
#       input_topic.submit(articletitle,input_topic,output_title,output_title)
#       input_title.click(outlinecreator,output_title,output_blog_sections)
#   complete_blog_btn = gr.Button("Generate Blog")
      
#   complete_blog_btn.click(paragraphgeneration,[output_title,output_blog_sections],[complete_blog_text])
 
# demo.launch(debug=True)
def higherorderqa(paragraph:str):
  output = ai21.Completion.execute(
  model="j2-large",
  prompt=f"""
  for the following paragraph generate four questions
  Paragraph:
  Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. 
Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  
A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.  A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency.
Questions:
1. How did Elon Musk's tweets cause a stir in the digital currency market?
Answer: Musk tweeted that Tesla would not accept payments in Bitcoin due to environmental concerns and later tweeted about working with Dogecoin developers to improve transaction efficiency. This caused Bitcoin price to hit a two-month low while Dogecoin rallied around 20%. Musk has been openly supportive of Dogecoin but rarely tweets about Bitcoin. In a recent tweet, he stated that Tesla was concerned with the rapidly increasing use of fossil fuels for Bitcoin mining and transaction, thus suspending vehicle purchases using cryptocurrency.
2. How did Elon Musk's tweets affect the cryptocurrency market?
Answer: Elon Musk's tweet about Tesla not accepting payments in Bitcoin due to environmental concerns led the world's largest cryptocurrency hit to a two-month low, while Dogecoin rallied by about 20 percent.
3. What are some of the arguments for and against using cryptocurrencies as a solution to climate change, according to Elon Musk's tweets?
Answer: Elon Musk argues that while he strongly believes in cryptocurrency, it cannot drive an increase in fossil fuel use. On Twitter, some users praised his statement while others indicated that cryptocurrencies such as Dogecoin are here to stay and that previous assertions that cryptocurrency could become the world's future currency support this.
4. How did Elon Musk's tweet impact the value of Bitcoin, and what was his stance on cryptocurrency's role in adding to fossil fuel use?
Answer: Elon Musk's tweet triggered a downward spiral in Bitcoin's value due to his conviction that cryptocurrency cannot drive a massive increase in fossil fuel use, especially coal. The cryptocurrency has since stabilised.
================================
Paragraph:
Python is a high-level, interpreted programming language that is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation. It was created by Guido van Rossum in the late 1980s and named after the British comedy group Monty Python.
Python is known for its simplicity, readability, and ease of use. Its syntax is designed to be intuitive and concise, making it a popular choice for beginners and experts alike. Python's standard library provides a wide range of modules for tasks such as string processing, web development, and data manipulation, as well as support for many programming paradigms, including object-oriented, functional, and procedural programming.
One of the most popular frameworks for web development in Python is Django, which provides a high-level, model-view-controller (MVC) architecture for building scalable and secure web applications. Flask is another popular framework that is known for its simplicity and flexibility.
Python is also widely used in data science and machine learning due to its powerful libraries such as NumPy, Pandas, and Scikit-learn. These libraries provide efficient data manipulation and analysis tools, making Python a popular choice for data analysts and scientists.
In recent years, Python has become increasingly popular in the field of artificial intelligence and machine learning, thanks to libraries such as TensorFlow and PyTorch, which provide high-level abstractions for building and training deep neural networks.
Overall, Python's versatility, ease of use, and large community make it an excellent choice for a wide range of applications, from web development to scientific computing and machine learning.
Questions:
1. What are some programming tasks that Python is commonly used for?
Answer: Python is often used for web development, scientific computing, data analysis, artificial intelligence, and automation.
2. What are some popular uses of Python and what are some reasons that make it a popular choice for developers?
Answer: Python is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation. It is known for its simplicity, readability, and ease of use. Python's syntax is designed to be intuitive and concise which makes it a popular choice for beginners and experts alike. Its standard library provides a wide range of modules for tasks such as string processing, web development, and data manipulation. Python also supports many programming paradigms including object-oriented, functional, and procedural programming. Django and Flask are two popular frameworks for web development in Python which are known for their scalability, security, simplicity, and flexibility.
3. Why has Python become increasingly popular in the field of artificial intelligence and machine learning?
Answer: Python has incorporated powerful libraries such as Tensorflow and PyTorch that provide high-level abstraction for building and training deep neural networks.
4. What aspects have contributed to Python's popularity in the fields of data science, artificial intelligence, and machine learning?
Answer: Python's versatility, ease of use, and powerful libraries such as NumPy, Pandas, and Scikit-learn have made it a popular choice for data analysts and scientists. Additionally, deep learning libraries like Tensorflow and PyTorch provide high-level abstractions for building and training deep neural networks. Most importantly, a large community and a wide range of applications from web development to scientific computing MAks it the perfect choice in its respective fields.
================================
Paragraph:
Climate refers to the long-term patterns of temperature, precipitation, and other atmospheric conditions in a particular region or across the globe. It is a critical component of the Earth's system, and changes in climate can have profound impacts on natural ecosystems, human societies, and the global economy. In recent decades, scientists have observed significant changes in the Earth's climate, including rising temperatures, melting ice caps and glaciers, and more frequent extreme weather events like droughts, floods, and heat waves. These changes are largely driven by human activities, such as the burning of fossil fuels and deforestation, which release large amounts of greenhouse gases into the atmosphere and trap heat, leading to global warming. Addressing the challenges posed by climate change is one of the most pressing issues facing the world today, and will require concerted efforts from individuals, governments, and businesses around the world.
Questions:
1. What are some factors that contribute to climate change and how does it impact the world?
Answer: Factors such as burning of fossil fuels and deforestation release greenhouse gases into the atmosphere and trap heat, leading to rising temperatures, melting ice caps and glaciers, and extreme weather events. Climate change has profound impacts on natural ecosystems, human societies, and the global economy, making it one of the most pressing issues facing the world today. Addressing this challenge will require concerted efforts from individuals, governments, and businesses around the world.
2. What factors have contributed to significant changes in the Earth's climate in recent decades?
Answer: Human activities such as burning of fossil fuels and deforestation have released large amounts of greenhouse gases into the atmosphere, trapping heat and leading to global warming. These changes in climate include rising temperatures, melting ice caps and glaciers, and more frequent extreme weather events like droughts, floods and heat waves.
=============================
Paragraph:
Chanakya, also known as Kautilya or Vishnugupta, was a renowned ancient Indian economist, philosopher, statesman, and strategist. He lived in the 4th century BCE and is widely regarded as one of the most influential figures in Indian history. Chanakya is best known for his role as the chief advisor to Emperor Chandragupta Maurya, the founder of the Maurya Empire, one of the largest empires in ancient India.
Chanakya's life and teachings are chronicled in the Arthashastra, a text that is considered to be one of the world's oldest treatises on economics, politics, and governance. The Arthashastra contains detailed instructions on statecraft, including the management of the economy, foreign policy, military strategy, and the administration of justice. It also contains insights into human psychology and behavior, and provides guidance on how to achieve success and prosperity in life.
Chanakya is also remembered for his role in the downfall of the Nanda dynasty, which ruled over the Magadha region of India at the time. According to legend, Chanakya was outraged by the corruption and tyranny of the Nanda rulers, and set out to find a suitable candidate to overthrow them. He discovered Chandragupta, a young warrior from a humble background, and trained him in the art of warfare and statecraft. With Chanakya's guidance, Chandragupta was able to defeat the Nanda army and establish the Maurya Empire, which went on to dominate much of India for several centuries.
Today, Chanakya is widely revered in India as a symbol of wisdom, strategy, and governance. His teachings continue to influence Indian politics and culture, and he is considered to be a source of inspiration for many leaders and thinkers around the world.
Questions:
1. What is climate?
Answer: Climate refers to the long-term patterns of temperature, precipitation, and other atmospheric conditions in a particular region or across the globe. These patterns can have significant impacts on natural ecosystems, human societies, and the global economy.
2. What is climate and why is it important?
Answer: Climate refers to long-term weather patterns and is a crucial component of our planet's system. Changes in climate can have a significant impact on ecosystems, global economy, and human societies. Climate changes have already affected the Earth, including increased temperatures, melting of ice caps and glaciers, and more frequent extreme weather events.
3. What is causing climate change?
Answer: Climate change is largely caused by human activities such as the burning of fossil fuels and deforestation, which release large amounts of greenhouse gases into the atmosphere and trap heat, leading to global warming.
4. What causes climate change, and why is it a pressing issue?
Answer: Climate change is largely caused by human activities such as burning fossil fuels and deforestation that release greenhouse gases into the atmosphere and trap heat, leading to global warming. Climate change is a pressing issue that requires organized efforts from individuals, governments and businesses because it poses a risk to the environment and human livelihood now and in the future.
================================
Paragraph:
{paragraph}
Questions:
""",
  numResults=1,
  maxTokens=8191,
  temperature=1,
  topKReturn=0.55,
  topP=0.55,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["========="]
  )
  return output.completions[0]['data']['text']

def fillintheblanks(Paragraph:str):
  output = ai21.Completion.execute(
  model="j2-large",
  prompt=f"""
  Paragraph:
  Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. 
Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  
A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.  A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency.
fill in the blanks:
1. Elon Musk's company Tesla will not accept payments in ________ because of environmental concerns.
Answer: Bitcoin
2. Dogecoin's transaction efficiency is being improved by working with ________ of Dogecoin.
Answer: developers
3. Recently, Elon Musk has been showing more support for ________ than Bitcoin.
Answer: Dogecoin
4. Tesla suspending vehicle purchases using cryptocurrency due to concern over rapidly increasing use of ________ fuels for Bitcoin mining and transaction.
Answer: fossil
5. According to Musk, crypto can't drive a massive increase in ________ fuel use, especially coal.
Answer: fossil
6. Musk's statement triggered a downward spiral for Bitcoin value but the ________ has stabilised since.
Answer: cryptocurrency
=====================
Paragraph:
Python is a high-level, interpreted programming language that is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation. It was created by Guido van Rossum in the late 1980s and named after the British comedy group Monty Python.
Python is known for its simplicity, readability, and ease of use. Its syntax is designed to be intuitive and concise, making it a popular choice for beginners and experts alike. Python's standard library provides a wide range of modules for tasks such as string processing, web development, and data manipulation, as well as support for many programming paradigms, including object-oriented, functional, and procedural programming.
One of the most popular frameworks for web development in Python is Django, which provides a high-level, model-view-controller (MVC) architecture for building scalable and secure web applications. Flask is another popular framework that is known for its simplicity and flexibility.
Python is also widely used in data science and machine learning due to its powerful libraries such as NumPy, Pandas, and Scikit-learn. These libraries provide efficient data manipulation and analysis tools, making Python a popular choice for data analysts and scientists.
In recent years, Python has become increasingly popular in the field of artificial intelligence and machine learning, thanks to libraries such as TensorFlow and PyTorch, which provide high-level abstractions for building and training deep neural networks.
Overall, Python's versatility, ease of use, and large community make it an excellent choice for a wide range of applications, from web development to scientific computing and machine learning.
fill in the blanks:
1. Python is a high-level, ________ programming language that is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation.
Answer: interpreted
2. Python is named after the British comedy group ________.
Answer: Monty Python
3. Python’s syntax is designed to be ________ and concise, making it a popular choice for beginners and experts alike.
Answer: intuitive
4. Python is a programming language that provides modules for ________, data manipulation and web development.
Answer: string processing
5. Django is a popular framework for ________ in Python.
Answer: web development
6. Python is widely used in ________ and machine learning due to libraries such as NumPy, Pandas, and Scikit-learn.
Answer: data science
7. Python is a popular choice for data analysts and scientists because it provides efficient ________ manipulation and analysis tools.
Answer: data
=====================
Paragraph:
Climate refers to the long-term patterns of temperature, precipitation, and other atmospheric conditions in a particular region or across the globe. It is a critical component of the Earth's system, and changes in climate can have profound impacts on natural ecosystems, human societies, and the global economy. In recent decades, scientists have observed significant changes in the Earth's climate, including rising temperatures, melting ice caps and glaciers, and more frequent extreme weather events like droughts, floods, and heat waves. These changes are largely driven by human activities, such as the burning of fossil fuels and deforestation, which release large amounts of greenhouse gases into the atmosphere and trap heat, leading to global warming. Addressing the challenges posed by climate change is one of the most pressing issues facing the world today, and will require concerted efforts from individuals, governments, and businesses around the world.
fill in the blanks:
1. Climate refers to the long-term patterns of ________ in a particular region or across the globe.
Answer: temperature
2. Scientists have observed significant changes in the Earth's ________, including melting ice caps and glaciers.
Answer: climate
3. Changes in climate can have profound impacts on natural ________, human societies, and the global economy.
Answer: ecosystems
4. Climate change is largely driven by human activities, such as the burning of ________ and deforestation.
Answer: fossil fuels
5. The trapping of heat that leads to global warming is caused by large amounts of greenhouse gases released into the ________.
atmosphere
6. Addressing the challenges posed by climate change will require concerted efforts from ________s, governments, and businesses around the world.
Answer: individuals
=====================
Paragraph:
{Paragraph}
fill in the blanks:
""",
  numResults=1,
  maxTokens=8191,
  temperature=1,
  topKReturn=0,
  topP=1,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["===="]
  )
  return output.completions[0]['data']['text'].replace('  ','')


def mcq(paragraph:str):
  output = ai21.Completion.execute(
  model="j2-large",
  prompt=f"""
  by using the below paragraph, generate multiple choice questions
  Paragraph:
  Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. 
Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  
A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.  A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency.
Multiple choice questions:
1. Why did Elon Musk announce that Tesla will not accept payments in Bitcoin?
a) Technical issues
b) Environmental concerns
c) Legal issues
d) Lack of popularity
Answer: Environmental concerns
2. Which cryptocurrency rallied by about 20 percent after Elon Musk's tweet?
a) Dogecoin
b) Ethereum
c) Ripple
d) Bitcoin
Answer: Dogecoin
3. Which digital currency does Elon Musk frequently support through his tweets?
a) Litecoin
b) Bitcoin
c) Dogecoin
d) Ethereum
Answer: Dogecoin
4. What triggered a downward spiral for Bitcoin value?
a) Hackers manipulating the cryptocurrency market
b) Musk's statement about Tesla suspending vehicle purchases using cryptocurrency
c) A decline in global stock markets
d) A change in government regulations regarding cryptocurrency
Answer: Musk's statement about Tesla suspending vehicle purchases using cryptocurrency
5. What does Musk strongly believe in?
a) Fossil fuels
b) Wind turbines
c) Electric vehicles
d) Crypto
Answer: Crypto
6. Which fuel did Musk especially mention as a concern regarding the use of cryptocurrency?
a) Coal
b) Gasoline
c) Natural Gas
d) Solar energy
Answer: Coal
========================================
Paragraph:
Python is a high-level, interpreted programming language that is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation. It was created by Guido van Rossum in the late 1980s and named after the British comedy group Monty Python.
Python is known for its simplicity, readability, and ease of use. Its syntax is designed to be intuitive and concise, making it a popular choice for beginners and experts alike. Python's standard library provides a wide range of modules for tasks such as string processing, web development, and data manipulation, as well as support for many programming paradigms, including object-oriented, functional, and procedural programming.
One of the most popular frameworks for web development in Python is Django, which provides a high-level, model-view-controller (MVC) architecture for building scalable and secure web applications. Flask is another popular framework that is known for its simplicity and flexibility.
Python is also widely used in data science and machine learning due to its powerful libraries such as NumPy, Pandas, and Scikit-learn. These libraries provide efficient data manipulation and analysis tools, making Python a popular choice for data analysts and scientists.
In recent years, Python has become increasingly popular in the field of artificial intelligence and machine learning, thanks to libraries such as TensorFlow and PyTorch, which provide high-level abstractions for building and training deep neural networks.
Overall, Python's versatility, ease of use, and large community make it an excellent choice for a wide range of applications, from web development to scientific computing and machine learning.
Multiple choice questions:
1. Who created Python programming language?
a) Steve Jobs
b) Mark Zuckerberg
c) Bill Gates
d) Guido van Rossum
Answer: Guido van Rossum
2. What is Python known for?
a) Speed and performance
b) Complexity and difficulty
c) Large community size
d) Simplicity, readability, and ease of use
Answer: Simplicity, readability, and ease of use
3. What programming paradigms does Python support?
a) Relational and non-relational
b) Message passing and event-driven
c) Object-oriented, functional, and procedural
d) Imperative and declarative
Answer: Object-oriented, functional, and procedural
4. Which framework provides a high-level, model-view-controller (MVC) architecture for building scalable and secure web applications in Python?
a) Django
b) Node.js
c) Express
d) Flask
Answer: Django
5. What is the name of the Python library used for efficient data manipulation and analysis?
a) TensorFlow
b) NumPy
c) Scikit-learn
d) Pandas
Answer: Pandas
6. Which field has Python become increasingly popular in recent years due to libraries such as TensorFlow and PyTorch?
a) Cloud computing
b) Web development
c) Artificial intelligence and machine learning
d) Data science
Answer: Artificial intelligence and machine learning
========================================
Paragraph:
William Shakespeare was a renowned English poet, playwright, and actor born in 1564 in Stratford-upon-Avon. His birthday is most commonly celebrated on 23 April (see When was Shakespeare born), which is also believed to be the date he died in 1616.
Shakespeare was a prolific writer during the Elizabethan and Jacobean ages of British theatre (sometimes called the English Renaissance or the Early Modern Period). Shakespeare’s plays are perhaps his most enduring legacy, but they are not all he wrote. Shakespeare’s poems also remain popular to this day. 
Shakespeare's Family Life Records survive relating to William Shakespeare’s family that offer an understanding of the context of Shakespeare's early life and the lives of his family members. John Shakespeare married Mary Arden, and together they had eight children. John and Mary lost two daughters as infants, so William became their eldest child. John Shakespeare worked as a glove-maker, but he also became an important figure in the town of Stratford by fulfilling civic positions. His elevated status meant that he was even more likely to have sent his children, including William, to the local grammar school. 
William Shakespeare would have lived with his family in their house on Henley Street until he turned eighteen. When he was eighteen, Shakespeare married Anne Hathaway, who was twenty-six. It was a rushed marriage because Anne was already pregnant at the time of the ceremony. Together they had three children. Their first daughter, Susanna, was born six months after the wedding and was later followed by twins Hamnet and Judith. Hamnet died when he was just 11 years old.
Multiple choice questions:
1. What was William Shakespeare's profession?
a) poet, playwright and actor
b) teacher
c) doctor
d) scientist
Answer: poet, playwright and actor
2. On which date is William Shakespeare's birthday commonly celebrated?
a) 23 April
b) 25 December
c) 1 May
d) 4 July
Answer: 23 April
3. Which of the following ages is referred to as the English Renaissance or the Early Modern Period?
a) Industrial Revolution era
b) Victorian era
c) Elizabethan and Jacobean ages of British theatre
d) Middle Ages
Answer: Elizabethan and Jacobean ages of British theatre
4. What was John Shakespeare's profession?
a) blacksmith
b) glove-maker
c) weaver
d) farmer
Answer: glove-maker
5. How many children did John and Mary Shakespeare have?
a) six
b) eight
c) four
d) ten
Answer: eight
6. How old was Anne Hathaway when she married William Shakespeare?
a) forty-two
b) eighteen
c) twenty-six
d) thirty
Answer: twenty-six
========================================================================
Paragraph:
Chanakya, also known as Kautilya or Vishnugupta, was an ancient Indian philosopher, teacher, and advisor to the Mauryan emperor Chandragupta. He is considered one of the greatest scholars and strategists in Indian history. Chanakya was born in a Brahmin family in 4th century BCE in the Indian kingdom of Magadha. He studied in the famous ancient university of Takshashila, where he gained knowledge in various fields, including economics, politics, military strategy, and diplomacy.
After the destruction of Takshashila by the invading Greeks, Chanakya traveled to Pataliputra, the capital of Magadha, where he became an advisor to the king, Chandragupta Maurya. He helped Chandragupta to overthrow the powerful Nanda dynasty and establish the Mauryan empire.
Chanakya is best known for his treatise, the Arthashastra, which is a comprehensive guide to politics, economics, and governance. The Arthashastra covers a wide range of topics, including foreign policy, taxation, espionage, military strategy, and the role of the king in society.
Chanakya's philosophy emphasized the importance of strong leadership, discipline, and morality in governance. He believed that a good ruler must be able to balance the interests of the state and the people, and should use all means necessary to achieve his goals.
Chanakya's teachings and philosophy have had a lasting impact on Indian culture and politics. He is revered as a great thinker and strategist, and his ideas continue to be studied and applied by scholars and leaders around the world.
Multiple choice questions:
1. In which kingdom was Chanakya born in?
a) Magadha
b) Kalinga
c) Vedic states
d) Pandya
Answer: Magadha
2. Where did Chanakya study?
a) Harappan city of Dholavira
b) Nalanda
c) Bikrampur, Bangladesh
d) Takshashila
Answer: Takshashila
3. Who did Chanakya helped to establish the Mauryan empire?
a) Chandragupta Maurya
b) Asoka
c) Bindusara
d) Samudragupta
Answer: Chandragupta Maurya
4. What is Chanakya best known for?
a) His collection of recipes
b) His work as a painter
c) His philosophy on music
d) His treatise, the Arthashastra
Answer: His treatise, the Arthashastra
5. What topics are covered in the Arthashastra?
a) Politics, economics, and governance
b) Medicine and healthcare
c) Agriculture and farming
d) Art and literature
Answer: Politics, economics, and governance
===========================================================
Paragraph:
Albert Einstein was a German-born physicist who is widely considered to be one of the most influential scientists of the 20th century.Einstein is most famous for his theory of relativity, which fundamentally changed our understanding of space, time, and gravity.Einstein was awarded the Nobel Prize in Physics in 1921 for his work on theoretical physics and the discovery of the law of the photoelectric effect.
Einstein was also a pacifist and an advocate for nuclear disarmament, and he famously warned of the dangers of nuclear weapons in his later years.Einstein's famous equation E=mc², which relates energy and mass, is perhaps the most famous formula in all of science.
Einstein fled Nazi Germany in 1933 and eventually settled in the United States, where he continued his groundbreaking work in physics.Einstein's work also contributed to the development of quantum mechanics, which is now the foundation of modern physics.
In addition to his scientific contributions, Einstein was a skilled violinist and a passionate humanitarian.Einstein was a vocal advocate for civil rights, and he spoke out against racism and discrimination throughout his life.
Einstein's theories have been confirmed by numerous experiments over the years, and they continue to be the foundation of our understanding of the universe.Einstein's work has also had practical applications, including the development of GPS technology and medical imaging techniques.
Einstein's personal life was often tumultuous, with multiple marriages and strained relationships with his children.Einstein was a prolific writer, with dozens of books and articles on topics ranging from physics to philosophy.
Einstein was a member of the Royal Society, the American Academy of Arts and Sciences, and the National Academy of Sciences.Einstein was known for his unconventional thinking and his willingness to challenge established scientific dogma.
Einstein's contributions to science have had a profound impact on our understanding of the universe and our place in it.Einstein's legacy continues to inspire new generations of scientists, who are building on his work to push the boundaries of our knowledge even further.
Einstein's life and work continue to fascinate people around the world, and he remains one of the most celebrated and iconic figures in the history of science.
Multiple choice questions:
1. What is Albert Einstein famous for?
a) Discovery of DNA structure
b) Inventing the telephone
c) Theory of relativity
d) Atomic bomb
Answer: Theory of relativity
2. For what reason did Einstein receive the Nobel Prize in Physics?
a) Inventing the computer
b) Development of nuclear weapons
c) Discovery of gravity waves
d) Work on theoretical physics and discovery of photoelectric effect
Answer: Work on theoretical physics and disocevry of photoelectric effect
3. What equation is famously associated with Einstein?
a) E=mc²
b) F=ma
c) a²+b²=c²
d) pv=nrt
Answer: E=mc^2
4. In which country did Einstein settle after fleeing Nazi Germany?
a) France
b) United Kingdom
c) Spain
d) United States
Answer: United States
5. What is the foundation of modern physics?
a) Black holes
b) Classical mechanics
c) Quantum mechanics
d) Relativity theory
Answer: Quantum mechanics
========================================
Paragraph:
{paragraph}
Multiple choice questions:
""",
  numResults=1,
  maxTokens=8191,
  temperature=1,
  topKReturn=0,
  topP=1,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["===="]
  )
  return output.completions[0]['data']['text']


def truerfalse(paragraph:str):
  output=ai21.Completion.execute(
  model="j2-large",
  prompt=f"""
  by using the below paragraph generate true or false questions
  Paragraph:
  Python is a high-level, interpreted programming language that is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation. It was created by Guido van Rossum in the late 1980s and named after the British comedy group Monty Python.
Python is known for its simplicity, readability, and ease of use. Its syntax is designed to be intuitive and concise, making it a popular choice for beginners and experts alike. Python's standard library provides a wide range of modules for tasks such as string processing, web development, and data manipulation, as well as support for many programming paradigms, including object-oriented, functional, and procedural programming.
One of the most popular frameworks for web development in Python is Django, which provides a high-level, model-view-controller (MVC) architecture for building scalable and secure web applications. Flask is another popular framework that is known for its simplicity and flexibility.
Python is also widely used in data science and machine learning due to its powerful libraries such as NumPy, Pandas, and Scikit-learn. These libraries provide efficient data manipulation and analysis tools, making Python a popular choice for data analysts and scientists.
In recent years, Python has become increasingly popular in the field of artificial intelligence and machine learning, thanks to libraries such as TensorFlow and PyTorch, which provide high-level abstractions for building and training deep neural networks.
Overall, Python's versatility, ease of use, and large community make it an excellent choice for a wide range of applications, from web development to scientific computing and machine learning.
True or False:
1. Python is a low-level programming language.
True
False
A: False
2. Guido van Rossum is the creator of Python.
True
False
A: True
3. Python is not widely used for web development.
True
False
A: False
4. Python's syntax is difficult to read and understand.
True
False
A: False
5. Django is a low-level Python framework for web development.
True
False
A: False
6. Flask is a simple and flexible web development framework in Python.
True
False
A: True
7. Python is not widely used in data science and machine learning.
True
False
A: False
8. TensorFlow and PyTorch are libraries for building and training deep neural networks in Python.
True
False
A: True
=================================
Paragraph:
Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. 
Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  
A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.  A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency.
True or False:
1. Elon Musk announced that Tesla will no longer accept payments in Dogecoin.
True
False
A: False
2. The largest cryptocurrency hit a two-month high after Elon Musk's tweet announcing Tesla's decision about Bitcoin.
True
False
A: False
3. Elon Musk has often expressed his support for Dogecoin on his Twitter account in recent months.
True
False
A: True
4. Bitcoin's value increased by about 20 percent following Elon Musk's tweet about working with developers of Dogecoin
True
False
A: False
5. Elon Musk tweeted that Tesla will no longer accept cryptocurrency for vehicle purchases due to concerns about fossil fuel use.
True
False
A: True
6. Musk stated that he no longer believes in the potential of cryptocurrency.
True
False
A: False
7. The statement from Musk caused a decrease in the value of Bitcoin which has not recovered since.
True
False
A: False
8. Many Twitter users expressed agreement with Musk's tweet about cryptocurrency.
True
False
A: True
=================================
Paragraph:
Oscar Wilde was a renowned Irish writer, poet, and playwright who gained international fame during the late 19th century. Born on October 16, 1854, in Dublin, Ireland, he was the second child of Sir William Wilde and Jane Wilde, who was a prominent Irish nationalist writer. Oscar Wilde is best known for his witty and satirical plays, including "The Importance of Being Earnest" and "Lady Windermere's Fan," which are still performed today. His flamboyant personality and unconventional lifestyle also contributed to his fame, as he was known for his dandyism and homosexual relationships.
Wilde attended Trinity College in Dublin and later studied at Magdalen College, Oxford, where he became known for his wit and flamboyant style. He began his literary career as a poet, publishing his first collection of poetry, "Poems," in 1881. In 1884, he married Constance Lloyd, and they had two sons together. However, Wilde's homosexuality would ultimately lead to the breakdown of their marriage.
Wilde's plays, which often featured social commentary and sharp wit, were a huge success during his lifetime. However, his career took a drastic turn when he was accused of homosexuality, which was illegal at the time. Wilde was put on trial and eventually sentenced to two years of hard labor in prison. This experience left a profound impact on him, and he wrote his most famous work, "The Ballad of Reading Gaol," while in prison.
After his release from prison, Wilde went into self-imposed exile in France, where he continued to write but struggled financially. He died in Paris on November 30, 1900, at the age of 46, due to meningitis, which was exacerbated by an ear infection. Despite his tragic end, Wilde's legacy as a literary and cultural icon continues to inspire and entertain people around the world.
True or False:
1. Oscar Wilde was a writer, poet, and playwright from Scotland.
True
False
A: False
2. Oscar was the second-born child in his family.
True
False
A: True
3. His most famous works include "The Importance of Being Earnest" and "Hamlet."
True
False
A: False
4. Wilde was known for his unconventional lifestyle and had many heterosexual relationships.
True
False
A: False
5. Wilde's literary career began with the publication of his first play, not his first collection of poetry.
True
False
A: False
6. Wilde's marriage to Constance Lloyd was unsuccessful because of his infidelity.
True
False
A: False
=================================
Paragraph:
{paragraph}
True or False:
""",
  numResults=1,
  maxTokens=8191,
  temperature=1,
  topKReturn=0,
  topP=1,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["==="]
 )
  return output.completions[0]['data']['text']

def faq(paragraph:str):
  output = ai21.Completion.execute(
  model="j2-large",
  prompt=f"""
  by using the below paragraph generate FAQ
  Paragraph:
  Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. 
Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency.  
A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since.  A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency.
FAQ:
1. What did Elon Musk say about Tesla and Bitcoin?
Answer: Elon Musk tweeted that Tesla would not accept payments in Bitcoin due to the cryptocurrency's environmental impact.
2. What is Elon Musk's stance on Bitcoin?
Answer: Elon Musk stated that his company Tesla will not accept payments in Bitcoin due to environmental concerns. This resulted in Bitcoin hitting a two-month low.
3. What prompted Tesla to suspend vehicle purchases using Bitcoin?
Answer: In a tweet by Elon Musk, Tesla expressed concerns about the rapidly increasing use of fossil fuels for Bitcoin mining and transactions. As a result, the company decided to suspend vehicle purchases using the cryptocurrency.
4. What is Tesla's stance on purchasing vehicles using Bitcoins?
Answer: Tesla has suspended vehicle purchases using cryptocurrency due to the rapidly increasing use of fossil fuels in Bitcoin mining and transactions, which the company is concerned about.
=======================================
Paragraph:
Python is a high-level, interpreted programming language that is widely used for web development, scientific computing, data analysis, artificial intelligence, and automation. It was created by Guido van Rossum in the late 1980s and named after the British comedy group Monty Python.
Python is known for its simplicity, readability, and ease of use. Its syntax is designed to be intuitive and concise, making it a popular choice for beginners and experts alike. Python's standard library provides a wide range of modules for tasks such as string processing, web development, and data manipulation, as well as support for many programming paradigms, including object-oriented, functional, and procedural programming.
One of the most popular frameworks for web development in Python is Django, which provides a high-level, model-view-controller (MVC) architecture for building scalable and secure web applications. Flask is another popular framework that is known for its simplicity and flexibility.
Python is also widely used in data science and machine learning due to its powerful libraries such as NumPy, Pandas, and Scikit-learn. These libraries provide efficient data manipulation and analysis tools, making Python a popular choice for data analysts and scientists.
In recent years, Python has become increasingly popular in the field of artificial intelligence and machine learning, thanks to libraries such as TensorFlow and PyTorch, which provide high-level abstractions for building and training deep neural networks.
Overall, Python's versatility, ease of use, and large community make it an excellent choice for a wide range of applications, from web development to scientific computing and machine learning.
FAQ:
1. What is Python and what is it used for?
Answer: Python is a high-level, interpreted programming language that is regularly used for a range of applications such as web development, data analysis, scientific computing, artificial intelligence and automation. Its simplicity, readability, and ease of use makes it a popular choice for both beginners and experts. Created by Guido van Rossum, named after Monty Python's Flying Circus, this language presents interpretations which prove intuitive and concise.
2. What is Python?
Answer: Python is a high-level, interpreted programming language known for its simplicity, readability, and ease of use. It is used in various fields, including web development, scientific computing, data analysis, artificial intelligence, and automation. Python was created by Guido van Rossum in the late 1980s and named after the British comedy group Monty Python.
3. What is Python used for?
Answer: Python is a versatile programming language that has many applications. It is commonly used for tasks such as string processing, web development, data manipulation, and data science.
4. What kind of support does Python's standard library provide?
Answer: Python's standard library provides several modules for string processing, web development, and data manipulation. Additionally, it supports several programming paradigms including object-oriented, functional, and procedural programming.
5. What makes Python a popular choice for data analysis and manipulation?
Answer: Python's popularity in the field of data analysis stems from its efficient data manipulation and analysis tools available in libraries such as Pandas, Numpy, and Matplotlib.
6. What kind of tools does Python offer for data analysis?
Answer: Python offers efficient data manipulation and analysis tools, making it a popular choice among data analysts and scientists.
=======================================
Paragraph:
Climate refers to the long-term patterns of temperature, precipitation, and other atmospheric conditions in a particular region or across the globe. It is a critical component of the Earth's system, and changes in climate can have profound impacts on natural ecosystems, human societies, and the global economy. In recent decades, scientists have observed significant changes in the Earth's climate, including rising temperatures, melting ice caps and glaciers, and more frequent extreme weather events like droughts, floods, and heat waves. These changes are largely driven by human activities, such as the burning of fossil fuels and deforestation, which release large amounts of greenhouse gases into the atmosphere and trap heat, leading to global warming. Addressing the challenges posed by climate change is one of the most pressing issues facing the world today, and will require concerted efforts from individuals, governments, and businesses around the world.
FAQ:
1. What is climate?
Answer: Climate refers to the long-term patterns of temperature, precipitation, and other atmospheric conditions in a particular region or across the globe. These patterns can have significant impacts on natural ecosystems, human societies, and the global economy.
2. What is climate and why is it important?
Answer: Climate refers to long-term weather patterns and is a crucial component of our planet's system. Changes in climate can have a significant impact on ecosystems, global economy, and human societies. Climate changes have already affected the Earth, including increased temperatures, melting of ice caps and glaciers, and more frequent extreme weather events.
3. What is causing climate change?
Answer: Climate change is largely caused by human activities such as the burning of fossil fuels and deforestation, which release large amounts of greenhouse gases into the atmosphere and trap heat, leading to global warming.
4. What causes climate change, and why is it a pressing issue?
Answer: Climate change is largely caused by human activities such as burning fossil fuels and deforestation that release greenhouse gases into the atmosphere and trap heat, leading to global warming. Climate change is a pressing issue that requires organized efforts from individuals, governments and businesses because it poses a risk to the environment and human livelihood now and in the future.
=======================================
Paragraph:
Oscar Wilde was a renowned Irish writer, poet, and playwright who gained international fame during the late 19th century. Born on October 16, 1854, in Dublin, Ireland, he was the second child of Sir William Wilde and Jane Wilde, who was a prominent Irish nationalist writer. Oscar Wilde is best known for his witty and satirical plays, including "The Importance of Being Earnest" and "Lady Windermere's Fan," which are still performed today. His flamboyant personality and unconventional lifestyle also contributed to his fame, as he was known for his dandyism and homosexual relationships.
Wilde attended Trinity College in Dublin and later studied at Magdalen College, Oxford, where he became known for his wit and flamboyant style. He began his literary career as a poet, publishing his first collection of poetry, "Poems," in 1881. In 1884, he married Constance Lloyd, and they had two sons together. However, Wilde's homosexuality would ultimately lead to the breakdown of their marriage.
Wilde's plays, which often featured social commentary and sharp wit, were a huge success during his lifetime. However, his career took a drastic turn when he was accused of homosexuality, which was illegal at the time. Wilde was put on trial and eventually sentenced to two years of hard labor in prison. This experience left a profound impact on him, and he wrote his most famous work, "The Ballad of Reading Gaol," while in prison.
After his release from prison, Wilde went into self-imposed exile in France, where he continued to write but struggled financially. He died in Paris on November 30, 1900, at the age of 46, due to meningitis, which was exacerbated by an ear infection. Despite his tragic end, Wilde's legacy as a literary and cultural icon continues to inspire and entertain people around the world.
FAQ:
1. Who was Oscar Wilde?
Answer: Oscar Wilde was a renowned Irish writer, poet, and playwright who gained international fame during the late 19th century.
2. What was Oscar Wilde's educational background?
Answer: Wilde attended Trinity College in Dublin and later studied at Magdalen College, Oxford.
3. Who is Oscar Wilde and what happened to him?
Answer: Oscar Wilde was an author and playwright in the 19th century. He was accused of homosexuality, which was illegal at the time, and put on trial. Wilde was sentenced to two years of hard labor in prison. 
=======================================
Paragraph:
{paragraph}
FAQ:
""",
  numResults=1,
  maxTokens=8191,
  temperature=1,
  topKReturn=0,
  topP=1,
  countPenalty={
    "scale": 0,
    "applyToNumbers": False,
    "applyToPunctuations": False,
    "applyToStopwords": False,
    "applyToWhitespaces": False,
    "applyToEmojis": False
  },
  frequencyPenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  presencePenalty={
      "scale": 0,
      "applyToNumbers": False,
      "applyToPunctuations": False,
      "applyToStopwords": False,
      "applyToWhitespaces": False,
      "applyToEmojis": False
  },
  stopSequences=["==="]
 )
  return output.completions[0]['data']['text']

with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber").set(
    loader_color="#ffbf00",
    slider_color="#ffbf00",
    button_primary_background_fill="*secondary_200",
    button_primary_background_fill_hover="*secondary_300",
    body_background_fill="*primary_200",
    body_background_fill_dark="*primary_300",
)) as demo:
  gr.Markdown("""<h1 style="color:black;font-family:'Brush Script MT', cursive;text-align:center;font-size: 50px">CognitioMaxima</h1>""")
  gr.Markdown("""<p style="color:black;font-family:monospace">Writing essays can be a daunting task, requiring extensive research and critical thinking skills. This app seeks to simplify the process by providing a user-friendly platform that streamlines essay writing. With this app, users can easily input their questions and generate corresponding essay content and quiz questions.
  The app's essay generator feature helps users to quickly and easily create essays by synthesizing information and summarizing key points. This saves users valuable time and allows them to focus on refining their ideas and arguments. In addition, the quiz question generator feature allows users to test their knowledge and identify areas where they may need further study.
  The app's intuitive interface and customizable settings make it easy for users to adjust the level of difficulty and scope of their quizzes and essays, ensuring that they are tailored to the user's specific needs and interests. With this app, users can improve their writing skills and expand their knowledge base in a fun and engaging way.</p>""")
  gr.HTML(value="<img src='https://i.ibb.co/xHhWQFq/cover-image.png' alt='Flow Diagram' width='1200' height='300'/>")
  with gr.Tab("Essay Generation"):
    with gr.Row():
      with gr.Column():
        input_topic = gr.Textbox(label="Topic (Be descriptive about it and press enter to generate a title)",placeholder="What the science says about global warming? (Press enter to continue)")
        input_title = gr.Button("Generate Blog Sections")         
      with gr.Column():
        output_title = gr.Textbox(label="Generated Title",placeholder="Artifical General Intelligence",interactive=False) 
        output_blog_sections = gr.Textbox(label="Blog Sections",placeholder="""1.What is artificial intelligence? 2.The history of AI 3.The limitations of AI 4.Artificial general intelligence 5.The future of AI 6.What to expect from AI Conclusion""",interactive=False) 
        complete_blog_text = gr.Textbox(label="Blog",interactive=False,placeholder=""" 1.What is artificial intelligence? Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and other animals. In computer science, AI research is defined as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". 2.The history of AI: Artificial intelligence has its origins in the question, "Can machines think?". Early research into artificial intelligence began in the late 1940s, and significant progress was made in the 1950s, with the introduction of programs that could solve simple problems and recognize words. 3.The limitations of AI: One of the limitations of AI is that it is currently limited in its ability to learn and adapt. This can make it difficult for AI systems to perform well in situations where there are large changes in data or parameters. Additionally, AI systems are prone to bias and generalization errors, which can make them less effective than humans in certain tasks. 4.Artificial general intelligence: Artificial general intelligence (AGI) is a hypothetical type of artificial intelligence that exhibits intelligent behavior across a broad range of tasks and applications. It is currently believed that general intelligence has not yet been achieved, although some experts believe that it may be possible to achieve it in the near future. 5.The future of AI: The future of AI is difficult to predict, but experts believe that it is likely that we will see significant advances in the technology in the coming years. It is likely that AI systems will become more accurate, more adaptive, and able to perform more complex tasks. Additionally, it is likely that we will see the development of AI systems that can interact with humans in a more natural and meaningful way. 6.What to expect from AI: As AI systems become more advanced, it is likely that we will see them being used in a variety of settings, including business, healthcare, and law enforcement. Additionally, it is likely that we will see the development of AI systems that can interact with humans in a more natural and meaningful way. """)
        input_topic.submit(articletitle,input_topic,output_title,output_title)
        input_title.click(outlinecreator,output_title,output_blog_sections)
    complete_blog_btn = gr.Button("Generate Blog")      
    complete_blog_btn.click(paragraphgeneration,[output_title,output_blog_sections],[complete_blog_text])
  with gr.Tab("Quiz mentor"):
    def execute_function(choice,x):
      
      print(f'Inside exceute {choice}')
      return {
          'MCQ': mcq(x),
          "Higher Order QA": higherorderqa(x),
          "FAQ": faq(x),
          "Fill in the Blanks": fillintheblanks(x),
          "TrueFalse": truerfalse(x)
          }.get(choice)
    def change(y:str,x:gr.SelectData):
      return {
          'MCQ': "you will get multiple choice questions",
          'Higher Order QA': "you will get Questions with simple answers ",
          'FAQ': "you will get Frequently Asked Questions with answers",
          'Fill in the Blanks': "you will get Fill in the Blank questions with answers",
          'TrueFalse': "you will get true or false type questions"
      }.get(x.value)

    
    with gr.Row():
      with gr.Column():
        
        options = gr.Radio(choices=["MCQ", "Higher Order QA","FAQ","Fill in the Blanks","TrueFalse"],label="Select Your choice")
        r=gr.Textbox(label="Description")
        paragraph_txt = gr.Textbox(label="Give me Passage")
        
        options.select(change,options,r)
      with gr.Column():
        op = gr.Textbox(label="Output")
        
    text_submit_btn = gr.Button("Submit")

    text_submit_btn.click(execute_function,[options,paragraph_txt],op)

    
demo.launch()
